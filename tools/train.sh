#!/usr/bin/env bash

#ps -ef | grep train.py | grep rongbo | grep -v grep | awk '{print "kill -9 "$2}' | sh

export NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PORT=30000 \
bash tools/dist_train.sh \
configs/L3_data_models/pointpillars/pointpillars_L3_vehicle_160e_p6000_pt8_v_025.py \
4 --work-dir work_dirs --extra-tag  pointpillars_L3_exp