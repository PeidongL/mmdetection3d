#!/usr/bin/env bash

#ps -ef | grep train.py | grep rongbo | grep -v grep | awk '{print "kill -9 "$2}' | sh

config=configs/L3_data_models/bevfusion/bevdepth_fusion.py
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PORT=29509

export NCCL_P2P_DISABLE=1 
bash tools/dist_train.sh \
$config \
4 --work-dir /mnt/intel/jupyterhub/mrb/work_dirs --extra-tag  auto_label_data