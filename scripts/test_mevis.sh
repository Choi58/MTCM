#!/usr/bin/env bash
set -x
cd ..

export MASTER_ADDR='localhost'
export MASTER_PORT='7666'
export WORLD_SIZE=1
export RANK=0

python train_net_dshmp.py \
    --config-file configs/refiner.yaml \
    --num-gpus 1 --dist-url auto --eval-only \
    MODEL.WEIGHTS pretrain/mtcm_model.pth \
    OUTPUT_DIR results/refiner DATASETS.TEST '("mevis_test",)'