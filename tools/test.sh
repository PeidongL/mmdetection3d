#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PORT=29509
config=work_dirs/L3_data_models/bevdepth_fusion/camera_fix_anchor_bug/bevdepth_fusion.py
pth=work_dirs/L3_data_models/bevdepth_fusion/camera_fix_anchor_bug/epoch_50.pth

export NCCL_P2P_DISABLE=1 
./tools/dist_test.sh $config $pth 4 --plus_eval --plot_result
