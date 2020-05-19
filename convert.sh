#!/bin/bash
output_dir=inceptionv3
export OMP_NUM_THREADS=16
python convert.py --model inceptionv3_res224 --epochs 20
