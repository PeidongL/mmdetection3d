import re, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fire
from typing import List


schema = 'thres,pr,rec'.split(',')
color='rgbcmyk'


# def main(fds='', save_plot_path=''):
#     print(fds)
#     print('--------------',save_plot_path)
#     fds = fds.split(',')
    
#     fns = filter(lambda x:(x.startswith('tpr') and not x.endswith('png')), os.listdir(fds[0]))
    
#     for i,fn in enumerate(fns):
#         plt.figure(i)
#         plt.title(fn)
#         for j,fd in enumerate(fds):
#             df1 = pd.read_csv(fd+"/%s"%fn, header=None, names=schema)
#             label_name =  fd.split('/')[-5] +'_' + fd.split('/')[-1].split('checkpoint_')[1].split('.')[0]
#             plt.plot(df1['rec'], df1['pr'], "*-%s"%color[j], label=label_name.replace('prefusion_',''))
#         plt.legend(loc="lower left")
#         plt.savefig(os.path.join(save_plot_path, '%s.png'%fn))
        
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
            label_name = exps[j].split('/')[-3] + '_e' + str(indexs[j])
            plt.plot(tpr['rec'], tpr['pr'], "*-%s"%color[j], label=label_name)
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_plot_path, '%s.png'%plt_title))

if __name__ == '__main__':
    # fire.Fire(main)
    exps = ['/mnt/intel/jupyterhub/mrb/work_dirs/pcdet_bev_fusion_load_img_fea/lidar_only_80_epoch_25cm_4gpu_20221030-102547/eval',
        #   '/mnt/intel/jupyterhub/mrb/train_log/mm3d/pcdet_bev_fusion_load_img_fea/lss_80_epoch_25cm_4gpu_20221028-112316/eval',
          '/mnt/intel/jupyterhub/mrb/train_log/mm3d/pcdet_bev_fusion_load_img_fea/depthlss_80_epoch_25cm_4gpu_20221028-112703/eval',
          '/mnt/intel/jupyterhub/mrb/work_dirs/prefusion_L3_random_100/prefusion_L3_20221015-095649/eval',
          '/home/peidong.li/mmdetection3d/work_dirs/L3_data_models/pcdet/no_proj/pcdet_bev_fusion_height/2022-11-16T12-47-00/eval',
          '/home/peidong.li/mmdetection3d/work_dirs/L3_data_models/pcdet/debug/pcdet_bev_fusion_height/2022-11-11T16-07-56/eval',
          '/home/peidong.li/mmdetection3d/work_dirs/L3_data_models/pcdet/debug/pcdet_plus_fusion/2022-11-13T00-40-37/eval']
    indexs = [80, 80, 80, 80, 80, 80]
    distance = 100
    save_plot_path = '/home/peidong.li/mmdetection3d/work_dirs/L3_data_models/pcdet/debug/pcdet_bev_fusion_height/'
    main(exps, indexs, distance, save_plot_path)