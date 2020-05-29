#!/bin/bash
output_dir=mobilenetv2
export OMP_NUM_THREADS=16
#python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
#       --model mobilenetv2_width_mult14 --batch-size 128 --epochs 300 \
#       --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --cache-dataset \
#       --output-dir=${output_dir} --apex
#python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
#       --model mobilenetv2_width_mult14 --batch-size 128 --epochs 300 \
#       --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --cache-dataset \
#       --resume ${output_dir}/checkpoint.pth --output-dir=${output_dir} --apex
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
       --model mobilenetv2_width_mult14 --batch-size 128 --cache-dataset \
       --resume ${output_dir}/model_293.pth --output-dir=${output_dir} --apex --test-only
