#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 2002\
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} 
