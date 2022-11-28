#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.launch --nproc_per_node=$2 --master_port 2008 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
