###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################

from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from .modeling.vita_criterion import VitaSetCriterion
from .modeling.vita_matcher import VitaHungarianMatcher
from .modeling.transformer_decoder.dshmp_decoder_batch import DsHmpHierarchical
from transformers import BertModel, RobertaModel, RobertaTokenizerFast
import spacy
from .modeling.transformer_decoder.modules import ReferringTracker_RVOS, TemporalRefiner_RVOS
from einops import rearrange, repeat

@META_ARCH_REGISTRY.register()
class Refiner(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        test_topk_per_image: int,
        # vita
        vita_module: nn.Module,
        vita_criterion: nn.Module,
        num_frames: int,
        num_classes: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        freeze_detector: bool,
        test_run_chunk_size: int,
        test_interpolate_chunk_size: int,
        is_coco: bool,
        output_threshold: float,
        lang_backbone: nn.Module,
        feature_resizer: nn.Module,
        freeze_text_encoder: bool,
        tracker: nn.Module,
        refiner: nn.Module,
        #fuser: nn.Module
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.test_topk_per_image = test_topk_per_image

        # vita hyper-parameters
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.vita_module = vita_module
        self.vita_criterion = vita_criterion
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres

        if freeze_detector:
            for name, p in self.named_parameters():
                if not "vita_module" in name:
                    p.requires_grad_(False)
        self.test_run_chunk_size = test_run_chunk_size
        self.test_interpolate_chunk_size = test_interpolate_chunk_size

        self.is_coco = is_coco

        self.output_threshold = output_threshold
        self.resizer = feature_resizer
        # self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = lang_backbone
        print('whether freeze text encoder {}'.format(freeze_text_encoder))
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        self.tracker = tracker
        self.refiner = refiner
        #self.fuser = fuser

        # FREEZE
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)
        for p in vita_module.parameters():
            p.requires_grad_(False)
        for p in self.resizer.parameters():
            p.requires_grad_(False)
        for p in self.tracker.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        vita_deep_supervision = cfg.MODEL.VITA.DEEP_SUPERVISION

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sim_weight = cfg.MODEL.VITA.SIM_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            vita_last_layer_num=cfg.MODEL.VITA.LAST_LAYER_NUM,
        )

        # Vita
        num_classes = sem_seg_head.num_classes
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        vita_module = DsHmpHierarchical(cfg=cfg, in_channels=hidden_dim, aux_loss=vita_deep_supervision)

        # building criterion for vita inference
        vita_matcher = VitaHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        vita_weight_dict = {
            "loss_vita_ce": class_weight, "loss_vita_mask": mask_weight, "loss_vita_dice": dice_weight
        }
        if sim_weight > 0.0:
            vita_weight_dict["loss_vita_sim"] = sim_weight
            # vita_weight_dict["loss_con_sim"] = 0.05
            vita_weight_dict["loss_con_sim"] = 0.3

        if vita_deep_supervision:
            vita_dec_layers = cfg.MODEL.VITA.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(vita_dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in vita_weight_dict.items()})
            vita_weight_dict.update(aux_weight_dict)
        vita_losses = ["vita_labels", "vita_masks"]
        if sim_weight > 0.0:
            vita_losses.append("fg_sim")
            vita_losses.append("contrastive_sim")

        vita_criterion = VitaSetCriterion(
            num_classes,
            matcher=vita_matcher,
            weight_dict=vita_weight_dict,
            eos_coef=cfg.MODEL.VITA.NO_OBJECT_WEIGHT,
            losses=vita_losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            sim_use_clip=cfg.MODEL.VITA.SIM_USE_CLIP,
        )
        # text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_encoder = RobertaModel.from_pretrained('roberta-base')
        resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )
        tracker = ReferringTracker_RVOS(
            hidden_channel=hidden_dim,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=6, 
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
        refiner = TemporalRefiner_RVOS(
            hidden_channel=hidden_dim,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=6, 
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
        #fuser =  FrameVideoFuser(hidden_dim=hidden_dim, 
        #                         num_heads=cfg.MODEL.MASK_FORMER.NHEADS,
        #                         drop_out=0.1)
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.VITA.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # vita
            "vita_module": vita_module,
            "vita_criterion": vita_criterion,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": num_classes,
            "is_multi_cls": cfg.MODEL.VITA.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.VITA.APPLY_CLS_THRES,
            "freeze_detector": cfg.MODEL.VITA.FREEZE_DETECTOR,
            "test_run_chunk_size": cfg.MODEL.VITA.TEST_RUN_CHUNK_SIZE,
            "test_interpolate_chunk_size": cfg.MODEL.VITA.TEST_INTERPOLATE_CHUNK_SIZE,
            "is_coco": cfg.DATASETS.TEST[0].startswith("coco"),
            "output_threshold": cfg.MODEL.VITA.TEST_OUTPUT_THRESHOLD,
            "lang_backbone": text_encoder,
            "feature_resizer": resizer,
            "freeze_text_encoder": cfg.MODEL.VITA.FREEZE_TEXT_ENCODER,
            "tracker": tracker,
            "refiner": refiner,
            #"fuser": fuser
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, iterations=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            return self.train_model(batched_inputs, iterations)
        else:
            # NOTE consider only B=1 case.
            return self.inference(batched_inputs[0])

    def train_model(self, batched_inputs, iterations):
        images = []

        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        B = len(batched_inputs)
        BT = images.tensor.shape[0]
        T = BT // B

        lang_emb = [x['lang_tokens'].to(self.device) for x in batched_inputs]
        lang_emb = torch.cat(lang_emb, dim=0)

        # lang_feat, lang_feat_sentence, lang_mask = self.forward_text(lang_emb, device=self.device)
        lang_mask = [x['lang_mask'].to(self.device) for x in batched_inputs]
        lang_mask = torch.cat(lang_mask, dim=0)

        motion_map, static_map = self.proceesing_text_decoupling(batched_inputs[0]['sentence'])
        motion_map = motion_map.unsqueeze(0).to(self.device)
        static_map = static_map.unsqueeze(0).to(self.device)

        lang_feat_all = self.text_encoder(lang_emb, attention_mask=lang_mask) # B, Nl, 768
        lang_feat_sentence = lang_feat_all.last_hidden_state
        lang_feat = lang_feat_all.pooler_output
        lang_feat = self.resizer(lang_feat)

        lang_feat_fusion = self.resizer(lang_feat_sentence)

        motion_feat, static_feat = [], []
        for b in range(B):
            motion_map_b, static_map_b = self.proceesing_text_decoupling(batched_inputs[b]['sentence'])
            motion_map_b = motion_map_b.unsqueeze(0).to(self.device)
            static_map_b = static_map_b.unsqueeze(0).to(self.device)

            lang_feat_fusion_b = lang_feat_fusion[b][None, :]
            lang_feat_b = lang_feat[b][None, :]

            motion_feat_b = torch.cat([lang_feat_fusion_b[motion_map_b.bool()], lang_feat_b], dim=0)
            static_feat_b = torch.cat([lang_feat_fusion_b[static_map_b.bool()], lang_feat_b], dim=0)

            motion_feat.append(motion_feat_b)
            static_feat.append(static_feat_b)
        
        max_length_m = max(tensor.shape[0] for tensor in motion_feat)
        max_length_s = max(tensor.shape[0] for tensor in static_feat)

        motion_feat = torch.stack([F.pad(tensor, (0, 0, 0, max_length_m - tensor.shape[0]), value=0.) for tensor in motion_feat])
        static_feat = torch.stack([F.pad(tensor, (0, 0, 0, max_length_s - tensor.shape[0]), value=0.) for tensor in static_feat])
        
        """
        motion_map, static_map = self.proceesing_text_decoupling(batched_inputs[0]['sentence'])
        motion_map = motion_map.unsqueeze(0).to(self.device)
        static_map = static_map.unsqueeze(0).to(self.device)

        motion_feat = torch.cat([lang_feat_fusion[motion_map.bool()], lang_feat], dim=0)
        static_feat = torch.cat([lang_feat_fusion[static_map.bool()], lang_feat], dim=0)
        """

        lang_feat_sentence = lang_feat_sentence.permute(0, 2, 1)
        bs = len(images)

        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        #self.backbone.eval()
        #self.sem_seg_head.eval()
        #self.vita_module.eval()
        with torch.no_grad():
            features = self.backbone(images.tensor, lang_feat_sentence.repeat(T, 1, 1), lang_mask.repeat(T, 1, 1))

            BT = len(images) # batch * Frames
            T = self.num_frames if self.training else BT
            B = BT // T
            lang_mask = lang_mask.squeeze(-1)
            outputs, frame_queries, mask_features = self.sem_seg_head(features, lang_feat_fusion, lang_mask, 
                                                        static_feat=static_feat.permute(1,0,2).repeat(1,T,1))
            # frame_queries: [3, 12, 100, 256] ; mask_features: [12, 256, 128, 144]
            L = frame_queries.shape[0]
            mask_features = self.vita_module.vita_mask_features(mask_features)
            mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])

            vita_outputs, vita_frame_embds, vita_video_embds = self.vita_module(frame_queries, lang_feat_fusion, lang_mask, 
                                                                motion_feat.repeat(L,1,1), return_frame_query=True)
            if 'keep' in batched_inputs[0].keys():
                self.keep = batched_inputs[0]['keep']
            else:
                self.keep = False
            outputs_tracker = self.tracker(vita_frame_embds, mask_features, lang_feat_fusion.repeat(L,1,1), lang_mask, resume=self.keep)
            tracker_embds = outputs_tracker['pred_embds'] # l b c t q
            tracker_frame_embds = self.tracker.frame_forward(vita_frame_embds.flatten(0,1).permute(0,3,1,2), 
                                                             lang_feat_fusion.repeat(L*T,1,1), lang_mask)
        del vita_outputs, features, frame_queries
        """
        vita_frame_embds: l b t q c -> lb c t q
        vita_video_embds: l b q c -> lb t q c
        """
        tracker_embds = rearrange(tracker_embds[-1], '(l b) c t q -> l b c t q', l=L, b=B)
        #vita_video_embds = repeat(vita_video_embds, 'l b q c -> (l b) t q c', t=T)
        motion_feat = motion_feat.repeat(T,1,1) # bt n c
        outputs_refiner = self.refiner(tracker_embds, tracker_frame_embds.permute(0,2,3,1), mask_features, motion_feat.repeat(L,1,1), lang_mask)
        outputs_refiner['pred_fq_embed'] = outputs_tracker['pred_fq_embed']

        outputs_refiner['pred_logits'] = rearrange(outputs_refiner['pred_logits'].mean(dim=1), '(l b) q c -> l b q c', b=B) # b q c -> l b q c
        outputs_refiner['pred_masks'] = rearrange(outputs_refiner['pred_masks'], '(l b) q t h w -> l b q t h w', b=B)
        for i, aux_outputs in enumerate(outputs_refiner['aux_outputs']):
            outputs_refiner['aux_outputs'][i]['pred_logits'] = rearrange(aux_outputs['pred_logits'].mean(dim=1), '(l b) q c -> l b q c', b=B)
            outputs_refiner['aux_outputs'][i]['pred_masks'] = rearrange(aux_outputs['pred_masks'], '(l b) q t h w -> l b q t h w', b=B)
            outputs_refiner['aux_outputs'][i]['pred_fq_embed'] = outputs_tracker['aux_outputs'][i]['pred_fq_embed']
        ################################
        del outputs_tracker
        # mask classification target
        frame_targets, clip_targets = self.prepare_targets(batched_inputs, images)
        # bipartite matching-based loss
        losses, fg_indices = self.criterion(outputs, frame_targets)

        """
        vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], mask_features)
        for out in vita_outputs["aux_outputs"]:
            out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], mask_features)
        """

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        vita_loss_dict = self.vita_criterion(outputs_refiner, clip_targets, frame_targets, fg_indices)
        vita_weight_dict = self.vita_criterion.weight_dict

        if iterations < 2000:
            vita_weight_dict["loss_con_sim"] = 0.3 * iterations / 2000
        else:
            vita_weight_dict["loss_con_sim"] = 0.3
        for k in vita_loss_dict.keys():
            if k in vita_weight_dict:
                vita_loss_dict[k] *= vita_weight_dict[k]
        #losses.update(vita_loss_dict)
        return vita_loss_dict

    def proceesing_text_decoupling(self, text): # simple implementation
        doc = self.nlp(text)
        encoded_input = self.tokenizer(text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
        roberta_tokens = self.tokenizer.convert_ids_to_tokens(encoded_input.input_ids[0])
        roberta_offsets = encoded_input['offset_mapping'][0]

        motion_map = torch.zeros(40)
        static_map = torch.zeros(40)
        for index, (rt, offset) in enumerate(zip(roberta_tokens, roberta_offsets)):
            start, end = offset
            if rt != '<s>' and rt != '</s>':
                for token in doc:
                    if token.idx <= start and (token.idx + len(token.text)) >= end:
                        if token.pos_ == 'VERB' or token.pos_ == 'ADV':
                            try:
                                motion_map[index] = 1
                            except:
                                break
                            break
                        else:
                            try:
                                static_map[index] = 1
                            except:
                                break
                            break

        return motion_map, static_map

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        frame_gt_instances = []
        clip_gt_instances = []

        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            # print(_num_instance, '_num_instance')
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(self.device)
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else: #polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2, 3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long() # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()         # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()    # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                    "refer_id": targets_per_video["refer_id"]
                }
            )

            for f_i in range(self.num_frames):
                _cls = gt_classes_per_video.clone()
                _ids = gt_ids_per_video[:, f_i].clone()
                _mask = gt_masks_per_video[:, f_i].clone()

                valid = _ids != -1
                frame_gt_instances.append({
                    "labels": _cls[valid],
                    "ids": _ids[valid],
                    "masks": _mask[valid],
                })

        return frame_gt_instances, clip_gt_instances

    def inference(self, batched_inputs):
        frame_queries, mask_features = [], []
        num_frames = len(batched_inputs["image"])
        to_store = self.device if num_frames <= 36 else "cpu"

        lang_emb = batched_inputs['lang_tokens'].to(self.device)
        lang_mask = batched_inputs['lang_mask'].to(self.device)
        motion_map, static_map = self.proceesing_text_decoupling(batched_inputs['sentence'])
        motion_map = motion_map.unsqueeze(0).to(self.device)
        static_map = static_map.unsqueeze(0).to(self.device)

        lang_feat_all = self.text_encoder(lang_emb, attention_mask=lang_mask) # B, Nl, 768
        lang_feat_sentence = lang_feat_all.last_hidden_state
        lang_feat = lang_feat_all.pooler_output

        lang_feat_fusion = self.resizer(lang_feat_sentence)
        lang_feat_sentence = lang_feat_sentence.permute(0, 2, 1)
        lang_feat = self.resizer(lang_feat)

        motion_feat = torch.cat([lang_feat_fusion[motion_map.bool()], lang_feat], dim=0).unsqueeze(dim=0)
        static_feat = torch.cat([lang_feat_fusion[static_map.bool()], lang_feat], dim=0).unsqueeze(dim=0)
        lang_mask_ = lang_mask
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        for i in range(math.ceil(num_frames / self.test_run_chunk_size)):
            images = batched_inputs["image"][i*self.test_run_chunk_size : (i+1)*self.test_run_chunk_size]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            bs = images.tensor.shape[0]
            lang_feat_sentence_all = lang_feat_sentence.repeat(bs, 1, 1)
            lang_mask_all = lang_mask.repeat(bs, 1, 1)

            features = self.backbone(images.tensor, lang_feat_sentence_all, lang_mask_all)
            outputs, _frame_queries, _mask_features = self.sem_seg_head(features, lang_feat_fusion, lang_mask_, 
                                                                    static_feat=static_feat.permute(1,0,2).repeat(1,bs,1))

            _mask_features = self.vita_module.vita_mask_features(_mask_features)

            # BT is 1 as runs per frame
            frame_queries.append(_frame_queries[-1])    # T', fQ, C
            mask_features.append(_mask_features.to(to_store))  # T', C, H, W

        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        if batched_inputs['dataset_name'] == 'mevis':
            merge = True
        else:
            merge = False
        del outputs, images, batched_inputs

        frame_queries = torch.cat(frame_queries)[None]  # 1, T, fQ, C
        mask_features = torch.cat(mask_features)[None]     # T, C, H, W

        vita_outputs, vita_frame_embds, vita_video_embds = self.vita_module(frame_queries, lang_feat_fusion, lang_mask_, motion_feat, return_frame_query=True)

        motion_feat = motion_feat.repeat(num_frames,1,1)
        
        self.keep = False
        outputs_tracker = self.tracker(vita_frame_embds, mask_features, lang_feat_fusion, lang_mask, resume=self.keep)
        tracker_embds = outputs_tracker['pred_embds'] # l b c t q
        tracker_frame_embds = self.tracker.frame_forward(vita_frame_embds.flatten(0,1).permute(0,3,1,2), 
                                                        lang_feat_fusion.repeat(num_frames,1,1), lang_mask)
        del outputs_tracker

        tracker_embds = rearrange(tracker_embds[-1], '(l b) c t q -> l b c t q', l=1, b=1)
        outputs_refiner = self.refiner(tracker_embds, tracker_frame_embds.permute(0,2,3,1), mask_features, motion_feat, lang_mask)

        mask_cls = outputs_refiner["pred_logits"][-1].mean(dim=0) # cQ, K+1
        mask_pred = outputs_refiner["pred_masks"][-1]         # cQ, t, h, w

        del vita_outputs

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(20, 1).flatten(0, 1)
        idx = scores.squeeze(-1) > self.output_threshold
        if not merge:
            top_score, idx = scores.squeeze(-1).topk(1, sorted=False)
        scores_per_video = scores.squeeze(-1)[idx]
        labels_per_video = labels[idx]
        pred_masks = mask_pred[idx]
        pred_masks = retry_if_cuda_oom(F.interpolate)(
            pred_masks,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        ) # cQ, T, H, W
        pred_masks = pred_masks[:, :, : image_size[0], : image_size[1]]
        pred_masks = F.interpolate(
            pred_masks, size=(out_height, out_width), mode="bilinear", align_corners=False
        ) > 0.
        masks_per_video = pred_masks.sum(dim=0, keepdim=True).clamp(max=1)
        processed_results = {
            "image_size": (out_height, out_width),
            "pred_scores": scores_per_video.tolist(),
            "pred_labels": labels_per_video.tolist(),
            "pred_masks": masks_per_video.cpu(),
        }

        return processed_results


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class FrameVideoFuser(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, hidden_dim, num_heads, drop_out):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fuser = nn.MultiheadAttention(hidden_dim, num_heads, drop_out)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

    def forward(self, frame_embds, video_embds):
        """
        vita_frame_embds: l b t q c
        vita_video_embds: l b q c
        """
        L, B, T, fQ, C = frame_embds.shape
        frame_embds = rearrange(frame_embds, 'l b t q c -> (t q) (l b) c')
        video_embds = rearrange(video_embds, 'l b q c -> q (l b) c')

        fused_embds = self.fuser(query=frame_embds, key=video_embds, value=video_embds)[0] # tq lb c

        frame_embds = self.relu(self.fc(frame_embds))
        fused_embds = self.relu(self.fc2(fused_embds))

        output = fused_embds * frame_embds
        output = self.dropout(self.fc3(output))
        
        output = rearrange(output, '(t q) (l b) c -> l b t q c', t=T, b=B)
        return output