import re, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fire
from typing import List


schema = 'thres,pr,rec'.split(',')
        
def main(exps: List[str],indexs: List[int],distance: int,save_plot_path: str):
    tpr_files = []
    os.makedirs(save_plot_path, exist_ok=True)
    for exp, index in zip(exps, indexs):
        cur_exp_tpr = []
        cur_tpr_files = list(filter(lambda x:(x.startswith('tpr') and not x.endswith('png')), os.listdir(exp)))
        cur_tpr_files.sort()
        for cur_tpr_file in cur_tpr_files:
            file_split = cur_tpr_file.split('_')
            cur_distance = int(file_split[1])
            cur_index = int(file_split[2])
            if cur_index==index and cur_distance==distance:
                cur_exp_tpr.append(cur_tpr_file)
        tpr_files.append(cur_exp_tpr)
        
    tpr_len = len(tpr_files[0])
    exp_len = len(exps)
    for i in range(tpr_len):
        plt.figure(i)
        plt_title = tpr_files[0][i].split('_')[0] + f'_{distance}'
        plt.title(plt_title)
        for j in range(exp_len):
            tpr_file = os.path.join(exps[j], tpr_files[j][i])
            tpr = pd.read_csv(tpr_file, header=None, names=schema)
            label_name = exps[j].split('/')[-3] + '/' + exps[j].split('/')[-2] + '_e' + str(indexs[j])
            # label_name = exps[j].split('/')[-3] + '/' + exps[j].split('/')[-2]
            plt.plot(tpr['rec'], tpr['pr'], "*-", label=label_name)
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_plot_path, '%s.png'%plt_title))

if __name__ == '__main__':
    # fire.Fire(main)
    root_dir = '/mnt/intel/jupyterhub/mrb/work_dirs'
    exps = [
    # "L4_data_models/lss_bevfusion_48x96/lss_bevfusion_lidar/eval",
    # "L4_data_models/lss_bevfusion_48x96/lss_bevfusion_fusion/eval",

    # "L4_data_models/bevfusion_48x96/bevdet2.0_lidar_only/eval",
    # "L4_data_models/bevfusion_48x96/bevdet2.0_fusion/eval",
    # "L4_data_models/pcdet_L4_bev_fusion_height_48x96/bevheight_fusion/eval",
    # # camera only
    # "L4_data_models/pcdet_L4_bev_fusion_height_48x96/bevheight_camera/eval",
    # "L4_data_models/pcdet_L4_bev_fusion_height_48x96/bevheight_epoch80/eval",
    ]
    
    distance = 100
    exps = {
    "L3_data_models/bevdet_fusion/lidar/eval": 800, # lidar 800

    # bevdet
    "L3_data_models/bevdet_fusion/bevdet2.0_fusion_fix_resize_bug/eval": 800, # fusion
    "L3_data_models/bevdet_fusion/camera_fea_45x80_depth_1x100/eval": 800, # camera 0-100m
    # "L3_data_models/bevdet_fusion/camera_fix_img_resize_bug_resume/eval": 60, # camera 0-72m
    
    # bev depth 
    # "L3_data_models/bevdepth_fusion/camera_resume/eval": 430, # camera 0-72， 没结果
    "L3_data_models/bevdepth_fusion/camera_retrain_0_100/eval": 800, # 800
    "L3_data_models/bevdepth_fusion/fusion_resume/eval": 560, # 560
    
    "L3_data_models/pcdet_bev_fusion_height_pure/camera/eval": 160, #160
    
    }
    
    exps = {
    "L3_data_models/bevdet_fusion/camera_fea_45x80_depth_1x100/eval": 800, # camera 0-100m
        "L3_data_models/bevdet_fusion/camera_fix_resize_feature_bug/eval": [x*100 for x in range(8,9)],
        "L3_data_models/bevdepth_fusion/bevdepth_camera_fix_resize_cam_fea/eval":  [x*100 for x in range(5,7)]
        
    }

    
    
    save_plot_path = os.path.split(__file__)[0]
    indexs = []
    exps_list = []
    for name, val in exps.items():
        index = val if isinstance(val, list) else [val]
        for idx in index: 
            exps_list.append(os.path.join(root_dir, name))
            indexs.append(idx)
        
    main(exps_list, indexs, distance, save_plot_path)