# Copyright (c) Plus. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from ..builder import NECKS
from torch.cuda.amp.autocast_mode import autocast
from mmcv.cnn import xavier_init, constant_init
from mmdet3d.ops import bev_pool
from mmcv.cnn import build_conv_layer
from mmdet.models.utils import LearnedPositionalEncoding
from mmdet.models.backbones.resnet import BasicBlock
from .. import builder
import numpy as np
import torch.nn.functional as F
from .view_transformer import LSSViewTransformer, LSSViewTransformerBEVDepth, DepthNet, ASPP

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

class CBAM_Block(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )
    def forward(self, x):
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        return x * self.att(result)

class IHRLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 only_use_height,
                 num_cams,
                 num_points=4,):
        super().__init__()
        
        self.only_use_height = only_use_height
        if only_use_height:
            self.pos_offset = nn.Linear(out_channels, 1)
        else:
            self.pos_offset = nn.Linear(out_channels, 3)
        self.num_cams = num_cams
        self.num_points = num_points
        xavier_init(self.pos_offset, distribution='uniform', bias=0.)
        # self.img_offset = nn.Linear(out_channels, num_cams * 2)
        # constant_init(self.img_offset, 0.)
        self.attention_weights = nn.Linear(out_channels, num_cams)
        constant_init(self.attention_weights, val=0., bias=0.)
        self.output_proj = nn.Linear(out_channels, out_channels)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self.dropout = nn.Dropout(p=0.0)

        self.vtransform = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            SE_Block(out_channels),
            CBAM_Block(kernel_size=7),
            # nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True),
            # SE_Block(out_channels),
            # CBAM_Block(kernel_size=7),
        )
        # self.seblock = SE_Block(out_channels)
        # self.cbam = CBAM_Block(kernel_size=7)

    def forward(self, inputs, bev_queries, bev_pos, reference_points, point_cloud_range, 
                lidar2img, img_aug, image_shapes, bev_h, bev_w, depth_prob=None, prob_range=None):
        #B, HW, 3
        inp_residual = bev_queries
        bev_queries = bev_queries + bev_pos
        bs, num_query = bev_queries.size()[:2]
        pos_offset = self.pos_offset(bev_queries)
        if self.only_use_height:
            # reference_points = torch.cat((reference_points, pos_offset.sigmoid().unsqueeze(1)), -1)
            reference_points[...,2:3] = inverse_sigmoid(reference_points[...,2:3].clone())+pos_offset.unsqueeze(1)
            reference_points[...,2:3] = reference_points[...,2:3].sigmoid()
        else:
            reference_points += pos_offset

        attention_weights = self.attention_weights(bev_queries).view(bs, 1, num_query, self.num_cams, 1)
            
        # img_offset = self.img_offset(bev_queries) # B, HW, 2N
        output, bev_mask = feature_sampling(
            inputs, reference_points, depth_prob, prob_range, None, point_cloud_range, lidar2img, img_aug, image_shapes)

        output = torch.nan_to_num(output)
        bev_mask = torch.nan_to_num(bev_mask)
        attention_weights = attention_weights.sigmoid()*bev_mask
        output = output * attention_weights
        #  B,C ,Nq, N, D
        output = output.squeeze(-1).sum(-1)

        output = output.permute(0, 2, 1).contiguous()
        # output = self.output_proj(output) + inp_residual
        output = self.output_proj(output)
        output = self.dropout(output) + inp_residual
        output = output.permute(0, 2, 1).contiguous()

        # B, C, HW
        B, C = output.size()[:2]
        output = output.view(B, C, bev_h, bev_w)
        output = self.vtransform(output)
        # output = self.seblock(output) 
        # output = self.cbam(output) 
        output = output.flatten(2).permute(0, 2, 1).contiguous()
        # output = self.output_proj(output)

        return output, reference_points

@NECKS.register_module()
class HeightTrans(nn.Module):
    def __init__(self, grid_config=None, data_config=None, pc_range=None,
                 numC_input=512, numC_Trans=64, num_layer=3,
                 bev_h=128, bev_w=128, only_use_height=True, **kwargs):
        super(HeightTrans, self).__init__()
    
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.only_use_height = only_use_height
        self.num_layer = num_layer
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'depth': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        self.data_config = data_config
        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.pc_range = pc_range
        ihr_layers = []
        for i in range(self.num_layer):
            in_filters = self.numC_Trans
            out_filters = self.numC_Trans
            ihr_layers.append(
                IHRLayer(in_filters, out_filters, self.only_use_height, num_cams=6)
            )
        self.ihr_layers = nn.ModuleList(ihr_layers)
        self.img_feat = nn.Sequential(
            nn.Conv2d(self.numC_input,
                self.numC_Trans, 3, padding=1),
            nn.BatchNorm2d(self.numC_Trans),
            nn.ReLU(True),
            SE_Block(self.numC_Trans),
            CBAM_Block(kernel_size=7),
        )

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.query_embedding = nn.Embedding(self.bev_h*self.bev_w,
                                                self.numC_Trans)
        # nn.init.xavier_uniform_(self.query_embedding.weight)
        xavier_init(self.query_embedding, distribution='uniform', bias=0.)
        self.positional_encoding = LearnedPositionalEncoding(self.numC_Trans // 2, self.bev_h, self.bev_w)

    @staticmethod
    def get_reference_points(H, W, Z=8, bs=1, device='cuda', dtype=torch.float):
        """
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(1)
        return ref_2d

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x=self.img_feat(x)

        dtype = x.dtype

        ref_2d = self.get_reference_points(self.bev_h, self.bev_w, 1, 
        bs=B, device=x.device, dtype=dtype)
        
        ref_3d = torch.cat((ref_2d, torch.ones_like(ref_2d[..., :1])*0.55), -1)

        bev_queries = self.query_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(B, 1, 1)
        bev_mask = torch.zeros((B, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        #B, HW, C

        for ihr in self.ihr_layers:
            output, ref_points = ihr(x, bev_queries, bev_pos, ref_3d, self.pc_range, rots, trans, intrins, post_rots,
            post_trans, self.data_config['input_size'], self.bev_h, self.bev_w)
            bev_queries = output 
            ref_3d = ref_points
        output = output.permute(0, 2, 1).contiguous().view(B, -1, self.bev_h, self.bev_w)

        # #B, C , H ,W

        return output

def feature_sampling(feat, reference_points, depth_prob, prob_range, img_offset, pc_range, lidar2img, img_aug, image_shapes):
    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # print('height',reference_points[..., 2:3])
    # B, D, HW, 3
    reference_points = reference_points.permute(1, 0, 2, 3).contiguous()
    # print('reference_points', reference_points.size())
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1)
    # referenece_depth = reference_points[..., 2:3].clone()
    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 3, 4).repeat(D, 1, 1, num_query, 1, 1)
    img_aug = img_aug.view(
        1, B, num_cam, 1, 3, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points = lidar2img.to(torch.float32).matmul(reference_points.to(torch.float32).unsqueeze(-1)).squeeze(-1)
    
    # rots = rots.view(
    #     1, B, num_cam, 1, 3, 3).repeat(D, 1, 1, num_query, 1, 1)
    # trans = trans.view(
    #     1, B, num_cam, 1, 3).repeat(D, 1, 1, num_query, 1)
    # intrins = intrins.view(
    #     1, B, num_cam, 1, 3, 3).repeat(D, 1, 1, num_query, 1, 1)
    # post_rots = post_rots.view(
    #     1, B, num_cam, 1, 3, 3).repeat(D, 1, 1, num_query, 1, 1)
    # post_trans = post_trans.view(
    #     1, B, num_cam, 1, 3).repeat(D, 1, 1, num_query, 1)
    # bda = bda.view(
    #     1, B, num_cam, 1, 3, 3).repeat(D, 1, 1, num_query, 1, 1)

    # reference_points = torch.inverse(bda).matmul(reference_points.unsqueeze(-1)).squeeze(-1)

    # reference_points = reference_points.to(torch.float32) - trans.to(torch.float32)
    # reference_points = torch.inverse(rots.to(torch.float32)).matmul(reference_points.to(torch.float32).unsqueeze(-1))
    # # referenece_depth = reference_points.squeeze(-1)[..., 1:2].clone()

    # reference_points = torch.matmul(intrins.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)

    eps = 1e-5
    referenece_depth = reference_points[..., 2:3].clone()
    bev_mask = (reference_points[..., 2:3] > eps)

    reference_points_cam = torch.cat((reference_points[..., 0:2] / torch.maximum(
        reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])*eps), reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])), -1)
    if img_offset is not None:
        # img_offset = img_offset.view(B, num_query, num_cam, -1).permute(0, 2, 1, 3).contiguous().view(B*num_cam, num_query, 2).unsqueeze(-2)
        # img_offset[..., 0] /= image_shapes[1]
        # img_offset[..., 1] /= image_shapes[0]
        # reference_points_cam_lvl = inverse_sigmoid(reference_points_cam_lvl.clone()) + img_offset
        # reference_points_cam_lvl = reference_points_cam_lvl.sigmoid()
        # D, B, N, Nq, 3
        
        # img_offset = img_offset.sigmoid()
        img_offset = img_offset.view(B, num_query, num_cam, -1).permute(0, 2, 1, 3).unsqueeze(0)
        reference_points_cam[...,0:2] += img_offset
        # reference_points_cam[...,0] += img_offset[...,0]*image_shapes[1]
        # reference_points_cam[...,1] += img_offset[...,1]*image_shapes[0]
    reference_points_cam = torch.matmul(img_aug.to(torch.float32), 
                                       reference_points_cam.unsqueeze(-1).to(torch.float32)).squeeze(-1)   
    # reference_points_cam = torch.matmul(post_rots.to(torch.float32), 
    #                                    reference_points_cam.unsqueeze(-1).to(torch.float32))

    # reference_points_cam = reference_points_cam.squeeze(-1).to(torch.float32) + post_trans.to(torch.float32)
    reference_points_cam = reference_points_cam[..., 0:2]

    reference_points_cam[..., 0] /= image_shapes[1]
    reference_points_cam[..., 1] /= image_shapes[0]

    reference_points_cam = (reference_points_cam - 0.5) * 2

    bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    # D, B, N, num_query, 1
    if depth_prob is not None:
        # referenece_depth /= depth_prob.size()[1]
        referenece_depth = (referenece_depth-prob_range[0])/(prob_range[1]-prob_range[0])
        referenece_depth = (referenece_depth - 0.5) * 2
        bev_mask = (bev_mask & (referenece_depth > -1.0)
                    & (referenece_depth < 1.0))

    bev_mask = bev_mask.permute(1, 4, 3, 2, 0).contiguous()
    bev_mask = torch.nan_to_num(bev_mask)
    # bev_mask = bev_mask.view(B, num_cam, 1, num_query, D, 1).permute(0, 2, 3, 1, 4, 5)
    # B,1 ,Nq, N, D

    # sampled_feats = []
    # for lvl, feat in enumerate(mlvl_feats):[]
    BN, C, H, W = feat.size()
   
    # feat = feat.view(B*N, C, H, W)
    reference_points_cam_lvl = reference_points_cam.permute(1, 2, 3, 0, 4).contiguous().view(BN, num_query, D, 2).to(torch.float32)

    sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
    # print(sampled_feat.size())
    # print(sampled_feat.permute(0, 2, 3, 1)[12][8000][8].size())
    # print(sampled_feat.permute(0, 2, 3, 1)[...,:64])
    sampled_feat = sampled_feat.view(B, num_cam, C, num_query, D).permute(0, 2, 3, 1, 4).contiguous()
    # B,C ,Nq, N, D

    if depth_prob is not None:
        reference_points_cam = torch.cat([reference_points_cam, referenece_depth], dim=-1)
        reference_points_cam = reference_points_cam.permute(1, 2, 3, 0, 4).contiguous().view(B*num_cam, 1, num_query, 1, 3)
        depth_prob = depth_prob.unsqueeze(1)
        depth_prob = F.grid_sample(depth_prob, reference_points_cam)
        # print('prob',depth_prob.permute(0, 2, 3, 4, 1)[...,:1])
        depth_prob = depth_prob.view(B, num_cam, 1, num_query, 1).permute(0, 2, 3, 1, 4)
        depth_prob = torch.nan_to_num(depth_prob)
        bev_mask = depth_prob * bev_mask

    return sampled_feat, bev_mask

class DepthNetPlus(DepthNet):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True):
        super(DepthNetPlus, self).__init__(in_channels,
                                           mid_channels,
                                           context_channels,
                                           depth_channels,
                                           use_dcn=True,
                                           use_aspp=True)
        # self.context_conv = nn.Conv2d(
        #     mid_channels, context_channels, kernel_size=3, stride=1, padding=1)
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        # Upsample
        depth_conv_list.append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))
        depth_conv_list.append(nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1))
        depth_conv_list.append(nn.BatchNorm2d(mid_channels))
        depth_conv_list.append(nn.ReLU(inplace=True))
        depth_conv_list.append(nn.Conv2d(
            mid_channels, mid_channels, kernel_size=1, stride=1, padding=0))
        depth_conv_list.append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))
        depth_conv_list.append(nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1))
        depth_conv_list.append(nn.BatchNorm2d(mid_channels))
        depth_conv_list.append(nn.ReLU(inplace=True))

        # depth_conv_list.append(
        #     nn.ConvTranspose2d(mid_channels, mid_channels, 4, stride=2, padding=1))
        # depth_conv_list.append(nn.ReLU(True))
        # depth_conv_list.append(
        #     nn.ConvTranspose2d(mid_channels, mid_channels, 4, stride=2, padding=1))
        # depth_conv_list.append(nn.ReLU(True))

        # Depth pred
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return depth, context

@NECKS.register_module()
class LSSViewTransformerBEVHeightDepth(HeightTrans):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), downsample=16, **kwargs):
        super(LSSViewTransformerBEVHeightDepth, self).__init__(**kwargs)
        ds = torch.arange(*self.grid_config['depth'], dtype=torch.float)
        self.D = ds.shape[0]
        self.downsample = downsample
        self.create_grid_infos(**self.grid_config)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNetPlus(self.numC_input, self.numC_input,
                                      self.numC_Trans, self.D, **depthnet_cfg)
        self.img_feat = None

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.
        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
                                dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss
    
    @force_fp32()
    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape
        rots, trans, intrins, post_rots, post_trans, bda = input[1:7]
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        lidar2img_R = intrins.matmul(torch.inverse(rots)).matmul(torch.inverse(bda))
        lidar2img_t = -intrins.matmul(torch.inverse(rots)).matmul(trans.unsqueeze(-1))
        lidar2img = torch.cat((lidar2img_R, lidar2img_t), -1)
        img_aug = torch.cat((post_rots, post_trans.unsqueeze(-1)), -1)
        # depth_digit = depth.view(B, N, self.D, H, W)

        # img_feat = tran_feat.view(B, N, self.numC_Trans, H, W)
        dtype = input[0].dtype
        ref_2d = self.get_reference_points(self.bev_h, self.bev_w, 1, 
        bs=B, device=input[0].device, dtype=dtype)
        
        ref_3d = torch.cat((ref_2d, torch.ones_like(ref_2d[..., :1])*0.55), -1)

        bev_queries = self.query_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(B, 1, 1)
        bev_mask = torch.zeros((B, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        #B, HW, C

        for ihr in self.ihr_layers:
            output, ref_points = ihr(tran_feat, bev_queries, bev_pos, ref_3d, self.pc_range, lidar2img,
            img_aug, self.data_config['input_size'], self.bev_h, self.bev_w, depth, self.grid_config['depth'])
            bev_queries = output 
            ref_3d = ref_points
        output = output.permute(0, 2, 1).contiguous().view(B, -1, self.bev_h, self.bev_w)

        return output, depth

    def view_transform(self, input, depth, tran_feat):
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        depth_digit, tran_feat = self.depth_net(x, mlp_input)
        # depth_digit = x[:, :self.D, ...]
        # tran_feat = x[:, self.D:self.D +  self.numC_Trans, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)  
