import argparse
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from tools.export_onnx.vt_module import HeightTrans
from tools.export_onnx.rpn_module import RPN

import torch.onnx.symbolic_opset11 as sym_opset
import torch.onnx.symbolic_helper as sym_help
from torch.onnx import register_custom_op_symbolic
# import onnx_graphsurgeon as gs

def grid_sampler(g, input, grid, mode, padding_mode, align_corners): #long, long, long: contants dtype
    mode_i = sym_help._maybe_get_scalar(mode)
    paddingmode_i = sym_help._maybe_get_scalar(padding_mode)
    aligncorners_i = sym_help._maybe_get_scalar(align_corners)
    return g.op("GridSample", input, grid, interpolationmode_i=mode_i, paddingmode_i=paddingmode_i,
                aligncorners_i=aligncorners_i) #just a dummy definition for onnx runtime since we don't need onnx inference
sym_opset.grid_sampler = grid_sampler
register_custom_op_symbolic("::grid_sampler",grid_sampler,11)

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

        # ###################################### Convert vt model to ONNX ######################################
        # vt input: num_img_features, fH, fW
        front_left_feat = torch.ones(
            [1, img_feat_shapes[0], img_feat_shapes[1], num_img_features], dtype=torch.float32, device=torch.device('cuda:0'))
        front_right_feat = torch.ones(
            [1, img_feat_shapes[0], img_feat_shapes[1], num_img_features], dtype=torch.float32, device=torch.device('cuda:0'))
        side_left_feat = torch.ones(
            [1, img_side_feat_shapes[0], img_side_feat_shapes[1], num_img_features], dtype=torch.float32, device=torch.device('cuda:0'))
        side_right_feat = torch.ones(
            [1, img_side_feat_shapes[0], img_side_feat_shapes[1], num_img_features], dtype=torch.float32, device=torch.device('cuda:0'))
        lidar2img_left = torch.ones(
            [1, 1, 3, 4], dtype=torch.float32, device=torch.device('cuda:0'))
        lidar2img_right = torch.ones(
            [1, 1, 3, 4], dtype=torch.float32, device=torch.device('cuda:0'))
        lidar2img_side_left = torch.ones(
            [1, 1, 3, 4], dtype=torch.float32, device=torch.device('cuda:0'))
        lidar2img_side_right = torch.ones(
            [1, 1, 3, 4], dtype=torch.float32, device=torch.device('cuda:0'))
        vt_input = (front_left_feat,front_right_feat,side_left_feat,side_right_feat,
                    lidar2img_left,lidar2img_right, lidar2img_side_left, lidar2img_side_right)

        vt_input_names = ['front_left_feat','front_right_feat','side_left_feat','side_right_feat',
                    'lidar2img_left','lidar2img_right','lidar2img_side_left','lidar2img_side_right']

        vt_output_names = ['bev_features']
        output_onnx_file = 'height_trans_new.onnx'
        torch.onnx.export(vt_model, vt_input, output_onnx_file, verbose=True,
                          input_names=vt_input_names, output_names=vt_output_names, opset_version = 11, enable_onnx_checker=False)
        print("[SUCCESS] VT model is converted to ONNX.")

        # ###################################### Convert RPN model to ONNX ######################################
        # RPN input: NCHW
        rpn_input = torch.ones(
            [1, num_bev_features, grid_size[1], grid_size[0]], dtype=torch.float32, device=torch.device('cuda:0'))
        rpn_input_names = ['spatial_features']
        rpn_output_names = ['cls_preds', 'box_preds', 'dir_cls_preds']
        output_onnx_file = 'bev_rpn_new.onnx'
        torch.onnx.export(rpn_model, rpn_input, output_onnx_file, verbose=True,
                          input_names=rpn_input_names, output_names=rpn_output_names)
        print("[SUCCESS] RPN model is converted to ONNX.")


if __name__ == '__main__':
    main()
