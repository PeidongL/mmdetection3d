import argparse
from pathlib import Path

import numpy as np
import torch
import onnxruntime
import onnx

import pickle as pkl

from pcdet.config import cfg, cfg_from_yaml_file
from tools.export_onnx.vt_module import HeightTrans
from tools.export_onnx.rpn_module import RPN


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config of model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint to start from')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)

    return args, cfg


def main():
    # Load the configs
    args, cfg = parse_config()
    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    pillar_size = cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE']
    grid_size = [0, 0, 0]
    grid_size[0] = int(
        (point_cloud_range[3] - point_cloud_range[0]) / pillar_size[0])
    grid_size[1] = int(
        (point_cloud_range[4] - point_cloud_range[1]) / pillar_size[1])
    grid_size[2] = int(
        (point_cloud_range[5] - point_cloud_range[2]) / pillar_size[2])
    num_point_features = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES']
    max_num_pillars = cfg.DATA_CONFIG.DATA_PROCESSOR[2]['MAX_NUMBER_OF_VOXELS']['train']
    max_num_points_per_pillar = cfg.DATA_CONFIG.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL']
    num_bev_features = cfg.MODEL.MAP_TO_BEV['NUM_BEV_FEATURES']
    num_img_features = cfg.MODEL.MAP_TO_BEV['NUM_IMG_FEATURES']
    img_shapes = [540, 960]
    img_side_shapes = [540, 960]
    img_feat_shapes = [128, 240]
    img_side_feat_shapes = [48, 88]

    # Build up models
    vt_model = HeightTrans(model_cfg=cfg.MODEL.MAP_TO_BEV,
                           grid_size=np.array(grid_size),
                           img_shapes=img_shapes,
                           img_side_shapes=img_side_shapes,
                           point_cloud_range=point_cloud_range)

    rpn_model = RPN(model_cfg=cfg.MODEL,
                    num_class=len(cfg.CLASS_NAMES),
                    class_names=cfg.CLASS_NAMES,
                    grid_size=np.array(grid_size),
                    point_cloud_range=point_cloud_range)

    with torch.no_grad():
        checkpoint = torch.load(args.ckpt)
        model_state_disk = checkpoint['model_state']

        vt_update_model_state = {}
        rpn_update_model_state = {}
        for key, val in model_state_disk.items():
            if key[18:] in vt_model.state_dict() and vt_model.state_dict()[key[18:]].shape == model_state_disk[key].shape:
                vt_update_model_state[key[18:]] = val
            if key in rpn_model.state_dict() and rpn_model.state_dict()[key].shape == model_state_disk[key].shape:
                rpn_update_model_state[key] = val

        vt_state_dict = vt_model.state_dict()
        vt_state_dict.update(vt_update_model_state)
        vt_model.load_state_dict(vt_state_dict)
        vt_model.cuda()
        vt_model.eval()

        rpn_state_dict = rpn_model.state_dict()
        rpn_state_dict.update(rpn_update_model_state)
        rpn_model.load_state_dict(rpn_state_dict)
        rpn_model.cuda()
        rpn_model.eval()


        # ###################################### Validate VFE model ONNX/PyTorch ######################################
        print("Validating View Transformation ONNX model ...")
        fl_feat = np.load('/mnt/intel/jupyterhub/zhao.gong/workspace/4cam_comperasion/training/front_left_feature/000008.npy')
        fr_feat = np.load('/mnt/intel/jupyterhub/zhao.gong/workspace/4cam_comperasion/training/front_right_feature/000008.npy')
        sl_feat = np.load('/mnt/intel/jupyterhub/zhao.gong/workspace/4cam_comperasion/training/side_left_feature/000008.npy')
        sr_feat = np.load('/mnt/intel/jupyterhub/zhao.gong/workspace/4cam_comperasion/training/side_right_feature/000008.npy')

        front_left_feat = torch.from_numpy(fl_feat).cuda().float().permute(0,2,3,1)
        front_right_feat = torch.from_numpy(fr_feat).cuda().float().permute(0,2,3,1)
        side_left_feat = torch.from_numpy(sl_feat).cuda().float().permute(0,2,3,1)
        side_right_feat = torch.from_numpy(sr_feat).cuda().float().permute(0,2,3,1)

        calib_path = "/mnt/intel/jupyterhub/zhao.gong/workspace/4cam_comperasion/training/calib/000008.pkl"
        with open(calib_path, 'rb') as f:
            calib = pkl.load(f, encoding='latin1')
            # calib = ret['calib'][batch_idx]
            # print(calib)
            P1 = calib['P1']
            P2 = calib['P2']
            Tr_cam_to_imu = calib['Tr_cam_to_imu']
            Tr_imu_to_cam = np.linalg.pinv(Tr_cam_to_imu)

            P_side_left = calib['P_side_left']
            Tr_cam_to_imu_side_left = calib['Tr_cam_to_imu_side_left']
            Tr_imu_to_cam_side_left = np.linalg.pinv(Tr_cam_to_imu_side_left)
            P_side_right = calib['P_side_right']
            Tr_cam_to_imu_side_right = calib['Tr_cam_to_imu_side_right']
            Tr_imu_to_cam_side_right = np.linalg.pinv(Tr_cam_to_imu_side_right)

            xyz2left = np.matmul(P1, Tr_imu_to_cam)
            xyz2right = np.matmul(P2, Tr_imu_to_cam)
            xyz2side_left = np.matmul(P_side_left, Tr_imu_to_cam_side_left)
            xyz2side_right = np.matmul(P_side_right, Tr_imu_to_cam_side_right)
            
            xyz2left = np.expand_dims(np.expand_dims(xyz2left, 0), 0)
            xyz2right = np.expand_dims(np.expand_dims(xyz2right, 0), 0)
            xyz2side_left = np.expand_dims(np.expand_dims(xyz2side_left, 0), 0)
            xyz2side_right = np.expand_dims(np.expand_dims(xyz2side_right, 0), 0)

        lidar2img_left = torch.from_numpy(np.array(xyz2left)).cuda().float()
        lidar2img_right = torch.from_numpy(np.array(xyz2right)).cuda().float()
        lidar2img_side_left = torch.from_numpy(np.array(xyz2side_left)).cuda().float()
        lidar2img_side_right = torch.from_numpy(np.array(xyz2side_right)).cuda().float()

        vt_input = (front_left_feat,front_right_feat,side_left_feat,side_right_feat,
                    lidar2img_left,lidar2img_right, lidar2img_side_left, lidar2img_side_right)
        vt_input_names = ['front_left_feat','front_right_feat','side_left_feat','side_right_feat',
                    'lidar2img_left','lidar2img_right','lidar2img_side_left','lidar2img_side_right']
        vt_out_torch = vt_model(front_left_feat,front_right_feat,side_left_feat,side_right_feat,
                    lidar2img_left,lidar2img_right, lidar2img_side_left, lidar2img_side_right)
        
        vt_onnx_model = onnx.load(vt_model_file)
        onnx.checker.check_model(vt_onnx_model)
        onnx_vt_session = onnxruntime.InferenceSession(vt_model_file)
        onnx_vt_input_name1 = onnx_vt_session.get_inputs()[0].name
        onnx_vt_input_name2 = onnx_vt_session.get_inputs()[1].name
        onnx_vt_input_name3 = onnx_vt_session.get_inputs()[2].name
        onnx_vt_input_name4 = onnx_vt_session.get_inputs()[3].name
        onnx_vt_input_name5 = onnx_vt_session.get_inputs()[4].name
        onnx_vt_input_name6 = onnx_vt_session.get_inputs()[5].name
        onnx_vt_input_name7 = onnx_vt_session.get_inputs()[6].name
        onnx_vt_input_name8 = onnx_vt_session.get_inputs()[7].name
        onnx_vt_output_name = [onnx_vt_session.get_outputs()[0].name]
        vt_out_onnx = onnx_vt_session.run(onnx_vt_output_name, {onnx_vt_input_name1: vt_input[0].detach().cpu().numpy(), \
            onnx_vt_input_name2: vt_input[1].detach().cpu().numpy(),onnx_vt_input_name3: vt_input[2].detach().cpu().numpy(), \
            onnx_vt_input_name4: vt_input[3].detach().cpu().numpy(),onnx_vt_input_name5: vt_input[4].detach().cpu().numpy(), \
            onnx_vt_input_name6: vt_input[5].detach().cpu().numpy(),onnx_vt_input_name7: vt_input[6].detach().cpu().numpy(), \
            onnx_vt_input_name8: vt_input[7].detach().cpu().numpy()})
        
        bev_feat = np.load('/home/peidong.li/0.npy')
        # np.testing.assert_allclose(vt_out_torch.detach().cpu().numpy(), vt_out_onnx[0], rtol=1e-03, atol=1e-04)
        np.testing.assert_allclose(np.expand_dims(bev_feat, axis=0), vt_out_onnx[0], rtol=1e-03, atol=1e-04)
        np.save('/home/peidong.li/pth.npy',vt_out_torch.detach().cpu().numpy()) 
        np.save('/home/peidong.li/onnx.npy',vt_out_onnx[0]) 

        print("[SUCCESS] VT ONNX model validated.")

        # ####################################### Validate RPN model ONNX/PyTorch ######################################
        print("Validating RPN ONNX model ...")
        rpn_input = torch.ones(
            [1, num_bev_features, grid_size[1], grid_size[0]], dtype=torch.float32, device=torch.device('cuda:0'))
        rpn_out_torch = rpn_model(rpn_input)
        
        rpn_onnx_model = onnx.load(rpn_model_file)
        onnx.checker.check_model(rpn_onnx_model)
        onnx_rpn_session = onnxruntime.InferenceSession(rpn_model_file)
        onnx_rpn_input_name = onnx_rpn_session.get_inputs()[0].name
        onnx_rpn_output_name = [onnx_rpn_session.get_outputs()[0].name,
                                onnx_rpn_session.get_outputs()[1].name,
                                onnx_rpn_session.get_outputs()[2].name]
        rpn_out_onnx = onnx_rpn_session.run(onnx_rpn_output_name, {onnx_rpn_input_name: rpn_input.detach().cpu().numpy()})

        np.testing.assert_allclose(rpn_out_torch[0].detach().cpu().numpy(), rpn_out_onnx[0], rtol=1e-03, atol=1e-04)
        np.testing.assert_allclose(rpn_out_torch[1].detach().cpu().numpy(), rpn_out_onnx[1], rtol=1e-03, atol=1e-04)
        np.testing.assert_allclose(rpn_out_torch[2].detach().cpu().numpy(), rpn_out_onnx[2], rtol=1e-03, atol=1e-04)
        print("[SUCCESS] RPN ONNX model validated.")


if __name__ == '__main__':
    vt_model_file = "height_trans_new.onnx"
    rpn_model_file = "bev_rpn_new.onnx"
    main()
