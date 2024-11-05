set -x
cd ..

export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export WORLD_SIZE=1
export RANK=0

python train_net_dshmp.py \
    --config-file configs/dshmp_swin_tiny.yaml \
    --num-gpus 1 --dist-url auto \
    MODEL.WEIGHTS pretrain/model_final_86143f.pkl \
    OUTPUT_DIR results/dshmp

python train_net_dshmp.py \
    --config-file configs/tracker.yaml \
    --num-gpus 1 --dist-url auto \
    MODEL.WEIGHTS results/dshmp/model_final.pth \
    OUTPUT_DIR results/tracker

python train_net_dshmp.py \
    --config-file configs/refiner.yaml \
    --num-gpus 1 --dist-url auto \
    MODEL.WEIGHTS results/tracker/model_final.pth \
    OUTPUT_DIR results/refiner