import torch
from torch import nn
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from scipy.optimize import linear_sum_assignment
from einops import rearrange, repeat
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d
import torch.nn.functional as F
import math
class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class ReferringTracker_RVOS(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
    ):
        super(ReferringTracker_RVOS, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        #self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ref_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ref_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None

        self.sim_embed_frame = nn.Linear(hidden_channel, hidden_channel)
        self.sim_embed_clip = nn.Linear(hidden_channel, hidden_channel)
        self.contrastive_embed = MLP(hidden_channel, hidden_channel, hidden_channel, 3)

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        return

    def forward(self, frame_embeds, mask_features, lang_embeds, lang_mask, resume=False, return_indices=False):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        L, B, n_frame, n_q, _ = frame_embeds.shape

        frame_embeds = frame_embeds.flatten(0,1).permute(1,2,0,3)  # t, q, b, c
        lang_embeds = lang_embeds.permute(1,0,2)
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []
        frame_embeds = self.decoder_norm(frame_embeds) # 추가
        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]  # q b c
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                self.last_frame_embeds = single_frame_embeds
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        ret_indices.append(self.match_embds_batch(single_frame_embeds, single_frame_embeds))
                        output = self.transformer_ref_cross_attention_layers[j](
                            single_frame_embeds, single_frame_embeds, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_cross_attention_layers[j](
                            output, lang_embeds, memory_mask=None,
                            memory_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_ref_cross_attention_layers[j](
                            ms_output[-1], ms_output[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_cross_attention_layers[j](
                            output, output, memory_mask=None,
                            memory_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            else:
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        indices = self.match_embds_batch(self.last_frame_embeds, single_frame_embeds)
                        self.last_frame_embeds = single_frame_embeds[indices, torch.arange(bs)]
                        ret_indices.append(indices)
                        output = self.transformer_ref_cross_attention_layers[j](
                            single_frame_embeds[indices, torch.arange(bs)], self.last_outputs[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_cross_attention_layers[j](
                            output, lang_embeds,  memory_mask=None,
                            memory_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_ref_cross_attention_layers[j](
                            ms_output[-1], self.last_outputs[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_cross_attention_layers[j](
                            output, output,  memory_mask=None,
                            memory_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features) 

        outputs_class = outputs_class.transpose(2, 3).flatten(1,2)
        outputs_masks = outputs_masks.transpose(2, 3).flatten(1,2)
        #outputs_class (l, b, q, t, cls+1)
        #outputs_masks l b q t h w
        decoder_outputs = self.decoder_norm(outputs)

        pred_fq_embed = self.sim_embed_frame(frame_embeds) # t, q, b, c
        pred_fq_embed = rearrange(pred_fq_embed, 't q (l b) c -> l b t q c', l=L)

        pred_cq_embed = self.sim_embed_clip(decoder_outputs.mean(dim=0)) # d q lb c -> d lb q c
        pred_cq_embed = rearrange(pred_cq_embed, 'd q (l b) c -> d l b q c', l=L)

        pred_contrastive_embed = self.contrastive_embed(decoder_outputs.mean(dim=0)[-1])
        pred_contrastive_embed = rearrange(pred_contrastive_embed, 'q (l b) c -> l b q c', l=L)
        
        out = {
           'pred_logits': outputs_class[-1],  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks, pred_cq_embed, pred_fq_embed
           ),
           'pred_embds': decoder_outputs.permute(1,3,4,0,2),  # (t, l, q, b, c) -> (l, b, c, t, q)
           'pred_fq_embed': pred_fq_embed, # L B T Q C
           'pred_cq_embed': pred_cq_embed[-1], # L, B, cQ, C
           "pred_contrastive_embed": pred_contrastive_embed # L, B, cQ, C
        }

        if return_indices:
            return out, ret_indices
        else:
            return out

    def match_embds(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(ref_embds, cur_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = indices[1]
        return indices

    def match_embds_batch(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        bs = ref_embds.size(1)
        indices = []
        for b in range(bs):
            ref, cur = ref_embds[:, b, :].detach(), cur_embds[:, b, :].detach()
            ref = ref / (ref.norm(dim=1)[:, None] + 1e-6)
            cur = cur / (cur.norm(dim=1)[:, None] + 1e-6)
            cos_sim = torch.mm(ref, cur.transpose(0, 1))
            C = 1 - cos_sim

            C = C.cpu()
            C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

            indices_b = linear_sum_assignment(C.transpose(0, 1))
            indices_b = torch.tensor(indices_b[1], dtype=torch.long)
            indices.append(indices_b)
        indices = torch.stack(indices, dim=1)
        return indices

    @torch.jit.unused
    def _set_aux_loss(
            self, outputs_cls, outputs_mask, outputs_cq_embed, outputs_fq_embed
    ):
        return [{"pred_logits": a, "pred_masks": b, "pred_cq_embed": c, "pred_fq_embed": outputs_fq_embed}
                for a, b, c in zip(outputs_cls[:-1], outputs_mask[:-1], outputs_cq_embed[:-1])]

    def prediction(self, outputs, mask_features):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        L = outputs.shape[3] // mask_features.shape[0]
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        outputs_class = self.class_embed(decoder_output).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features.repeat(L,1,1,1,1).to(mask_embed.device))
        return outputs_class, outputs_mask
    
    def frame_forward(self, frame_embeds, lang_embeds, lang_mask):
        """
        only for providing the instance memories for refiner
        :param frame_embeds: the instance queries output by the segmenter, shape is (q, b, t, c)
        :return: the projected instance queries
        """
        bs, n_channel, n_frame, n_q = frame_embeds.size()
        frame_embeds = rearrange(frame_embeds, 'b c t q -> q (b t) c') # (q, bt, c)
        lang_embeds = lang_embeds.permute(1,0,2)

        for j in range(self.num_layers):
            if j == 0:
                output = self.transformer_ref_cross_attention_layers[j](
                    frame_embeds, frame_embeds, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_cross_attention_layers[j](
                    output, lang_embeds, memory_mask=None,
                    memory_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
            else:
                output = self.transformer_ref_cross_attention_layers[j](
                    output, output, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_cross_attention_layers[j](
                    output, lang_embeds, memory_mask=None,
                    memory_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
        output = self.decoder_norm(output)
        output = output.reshape(n_q, bs, n_frame, n_channel) # q b t c
        return output.permute(1, 3, 2, 0) # b c t q
    
class TemporalRefiner_RVOS(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        windows=5
    ):
        super(TemporalRefiner_RVOS, self).__init__()

        self.windows = windows

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_obj_self_attention_layers = nn.ModuleList()
        self.transformer_time_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for i in range(self.num_layers):
            self.transformer_time_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.conv_short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )
            self.conv_norms.append(nn.LayerNorm(hidden_channel))
            self.transformer_obj_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)
    
        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        self.activation_proj = nn.Linear(hidden_channel, 1)

        self.sim_embed_clip = nn.Linear(hidden_channel, hidden_channel)
        self.contrastive_embed = MLP(hidden_channel, hidden_channel, hidden_channel, 3)


    def forward(self, instance_embeds, frame_embeds, mask_features ,lang_feat, lang_mask):
        """
        :param instance_embeds: the aligned instance queries output by the tracker, shape is (b, c, t, q)
        :param frame_embeds: the instance queries processed by the tracker.frame_forward function, shape is (b, c, t, q)
        :param mask_features: the mask features output by the segmenter, shape is (b, t, c, h, w)
        :return: output dict, including masks, classes, embeds.
        """

        # instance_embeds: b c t q
        # frame_embeds: b t q c # b c t q
        # lang_feat: (b t) l c

        L = instance_embeds.shape[0]
        instance_embeds = instance_embeds.flatten(0,1)
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()

        outputs = []
        output = instance_embeds # b c t q
        frame_embeds = frame_embeds.permute(2,0,1,3).flatten(1, 2) # b t q c -> q (b t) c
        lang_feat = lang_feat.permute(1,0,2) # N, bt 256

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1)  # (t, b, q, c)
            output = output.flatten(1, 2)  # (t, bq, c)

            # do long temporal attention
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do short temporal conv
            output = output.permute(1, 2, 0)  # (bq, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)
            ).transpose(1, 2)
            output = output.reshape(
                n_batch, n_instance, n_channel, n_frames
            ).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)

            # do objects self attention
            output = self.transformer_obj_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )
            
            if i ==0:
                # do cross attention # 처음에만! lang_feat을 쓰도록
                output = self.transformer_cross_attention_layers[i](
                    output, lang_feat,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
            else:
                # do cross attention
                output = self.transformer_cross_attention_layers[i](
                    output, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0)  # (b, c, t, q)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2)  # (l, b, c, t, q) -> (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)

        decoder_outputs = self.decoder_norm(outputs)
        activation = self.activation_proj(decoder_outputs).softmax(dim=0) # t l q b c
        decoder_outputs = (decoder_outputs * activation).sum(dim=0) # d q lb c 

        pred_cq_embed = self.sim_embed_clip(decoder_outputs) # d q lb c -> d lb q c
        pred_cq_embed = rearrange(pred_cq_embed,'d q (l b) c -> d l b q c', l=L) 

        pred_contrastive_embed = self.contrastive_embed(decoder_outputs[-1])
        pred_contrastive_embed = rearrange(pred_contrastive_embed, 'q (l b) c -> l b q c', l=L)
        
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks, pred_cq_embed
           ),
           'pred_cq_embed': pred_cq_embed[-1],
           "pred_contrastive_embed": pred_contrastive_embed,
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_cq_embed):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b, "pred_cq_embed": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_cq_embed[:-1])
                ]

    def windows_prediction(self, outputs, mask_features, windows=5):
        """
        for windows prediction, because mask features consumed too much GPU memory
        """
        iters = outputs.size(0) // windows
        if outputs.size(0) % windows != 0:
            iters += 1
        outputs_classes = []
        outputs_masks = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_outputs = outputs[start_idx:end_idx]
            decoder_output = self.decoder_norm(clip_outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum(
                "lbtqc,btchw->lbqthw",
                mask_embed,
                mask_features[:, start_idx:end_idx].to(mask_embed.device)
            )
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        outputs_classes = self.pred_class(outputs_classes)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def pred_class(self, decoder_output):
        """
        fuse the objects queries of all frames and predict an overall score based on the fused objects queries
        :param decoder_output: instance queries, shape is (l, b, t, q, c)
        """
        T = decoder_output.size(2)

        # compute the weighted average of the decoder_output
        activation = self.activation_proj(decoder_output).softmax(dim=2)  # (l, b, t, q, 1)
        class_output = (decoder_output * activation).sum(dim=2, keepdim=True)  # (l, b, 1, q, c)

        # to unify the output format, duplicate the fused features T times
        class_output = class_output.repeat(1, 1, T, 1, 1)
        outputs_class = self.class_embed(class_output).transpose(2, 3)
        return outputs_class

    def prediction(self, outputs, mask_features):
        """
        :param outputs: instance queries, shape is (t, l, q, b, c)
        :param mask_features: mask features, shape is (b, t, c, h, w)
        :return: pred class and pred masks
        """
        if self.training:
            L = outputs.shape[3] // mask_features.shape[0]
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            outputs_class = self.pred_class(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features.repeat(L,1,1,1,1))
        else:
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=self.windows)
        return outputs_class, outputs_mask
