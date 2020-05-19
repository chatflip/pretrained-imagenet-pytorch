#!/bin/bash
output_dir=inceptionv3
export OMP_NUM_THREADS=16
#python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
#       --model inceptionv3_res224 --batch-size 256 --epochs 100 \
#       --lr 0.0045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.94 --cache-dataset \
#       --output-dir=${output_dir} --apex
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
       --model inceptionv3_res224 --batch-size 256 --epochs 100 \
       --lr 0.0045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.94 --cache-dataset \
       --resume ${output_dir}/checkpoint.pth --output-dir=${output_dir} --apex
#python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
#       --model inceptionv3_res224 --batch-size 256 --cache-dataset \
#       --resume ${output_dir}/model_20.pth --output-dir=${output_dir} --apex --test-only
