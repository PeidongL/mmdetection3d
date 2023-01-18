from collections import defaultdict
import cv2
import os
# img = cv2.imread('/mnt/intel/jupyterhub/mrb/datasets/L4E_wo_tele/L4E_origin_data/training/front_left_camera/007791.1618541811.398278.20210416T064550_j7-e0008_59_23to43.jpg')

# p=(100,10)
# cv2.circle(img, p, 1, (255,0,0), 2)
# print(img.shape)


# bp = [0,0,0]
# img[15:30, 100:115,1] = bp
# cv2.imwrite('aaa.png',img)

# pass


import pickle


def analysis_pkl(data_root, pkl_name):

    pkl_name=os.path.join(data_root, pkl_name)

    pkl_file = open(pkl_name, 'rb')
    data = pickle.load(pkl_file)
    shape_dict = defaultdict(int)
    for d in data:
        img_shape = d['image']['front_left_camera']['image_shape']
        shape_dict[img_shape]+=1
        # d['calib']['front_left_camera']['R0_rect'] ==d['calib']['front_right_camera']['R0_rect']
        # imgL = cv2.imread(root_path + d['image']['front_left_camera']['image_path'])
        # imgR = cv2.imread(root_path + d['image']['front_right_camera']['image_path'])
    print(data_root.split('/')[-2], shape_dict)

#l3
root_path= '/mnt/intel/jupyterhub/swc/datasets/L4E_extracted_data_1227/L4E_origin_data/'
analysis_pkl(root_path, 'Kitti_L4_lc_data_mm3d_infos_train_12192.pkl')


# l4
data_root = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/CN_L4_origin_data/'
hard_case_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/hard_case_origin_data/'
side_vehicle_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/side_vehicle_origin_data/'
under_tree_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/under_tree_origin_data/'
benchmark_root = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/CN_L4_origin_benchmark/'

pkls = [
    data_root, hard_case_data, side_vehicle_data, under_tree_data,
    benchmark_root]
for pkl in pkls:
    if 'benchmark' in pkl:
        analysis_pkl(pkl, 'Kitti_L4_data_mm3d_infos_val.pkl')
    else:
        analysis_pkl(pkl, 'Kitti_L4_data_mm3d_infos_train.pkl')
        



