from threading import currentThread
from typing import Tuple

import torch
from ..builder import NECKS
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import xavier_init, constant_init
from mmdet.models.utils import LearnedPositionalEncoding
from mmdet3d.models import apply_3d_transformation
from .depth_lss import DepthLSSTransform
from .base import BaseDepthTransform

__all__ = ["HeighTransform","HeightDepthTransform","HeightDepthFusion"]

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
                 num_cams=2,
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
        self.img_offset = nn.Linear(out_channels, num_cams * 2)
        # constant_init(self.img_offset, 0.)
        self.attention_weights = nn.Linear(out_channels, num_cams)
        constant_init(self.attention_weights, val=0., bias=0.)
        # self.output_proj = nn.Linear(out_channels, out_channels)
        # xavier_init(self.output_proj, distribution='uniform', bias=0.)
        # self.dropout = nn.Dropout(p=0.0)

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
        # self.stereo_consist = nn.Sequential(
        #     nn.Conv2d(out_channels * 2,
        #         out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(True),
        #     SE_Block(out_channels),
        # )

    def forward(self, inputs, bev_queries, bev_pos, reference_points, point_cloud_range, 
                lidar_to_img, bev_h, bev_w, depth_prob, img_metas, **kwargs):
        #B, HW, 3
        # inp_residual = bev_queries
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
            
        img_offset = self.img_offset(bev_queries) # B, HW, 2N
        # image_shapes=img_metas[0]['img_shape'][0]

        reference_points_3d, output, bev_mask = feature_sampling(
            inputs, reference_points, depth_prob, img_offset, point_cloud_range, lidar_to_img, img_metas)

        output = torch.nan_to_num(output)
        bev_mask = torch.nan_to_num(bev_mask)
        attention_weights = attention_weights.sigmoid()*bev_mask
        output = output * attention_weights
        #  B,C ,Nq, N, D
        # output = output.permute(0, 2, 1, 3, 4).flatten(2).permute(0, 2, 1).unsqueeze(-1)
        # output = self.stereo_consist(output).squeeze(-1)
        output = output.squeeze(-1).sum(-1)

        # output = output.permute(0, 2, 1).contiguous()
        # output = self.output_proj(output)
        # output = self.dropout(output) + inp_residual
        # output = output.permute(0, 2, 1).contiguous()

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
class HeighTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        bev_grid_map_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.only_use_height = True
        self.num_layer = 3
        self.pc_range = point_cloud_range
        ihr_layers = []
        for i in range(self.num_layer):
            in_filters = self.C
            out_filters = self.C
            ihr_layers.append(
                IHRLayer(in_filters, out_filters, self.only_use_height, num_cams=2)
            )
        self.ihr_layers = nn.ModuleList(ihr_layers)
        self.bev_h = bev_grid_map_size[0]
        self.bev_w = bev_grid_map_size[1]
        self.query_embedding = nn.Embedding(self.bev_h*self.bev_w, self.C)
        # nn.init.xavier_uniform_(self.query_embedding.weight)
        xavier_init(self.query_embedding, distribution='uniform', bias=0.)
        self.positional_encoding = LearnedPositionalEncoding(self.C // 2, self.bev_h, self.bev_w)

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

    @force_fp32()
    def forward(
        self, points, img_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics, **kwargs
    ):
        batch_size = len(points)
        num_cam = len(img_metas[0]['img_shape'])
  
        B, N, C, fH, fW = img_feats.shape
        img_feat = img_feats.view(B * N, C, fH, fW)

        dtype = img_feat.dtype
        ref_2d = self.get_reference_points(self.bev_h, self.bev_w, 1, 
        bs=batch_size, device=img_feat.device, dtype=dtype)
        
        ref_3d = torch.cat((ref_2d, torch.ones_like(ref_2d[..., :1])*0.55), -1)

        bev_queries = self.query_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        bev_mask = torch.zeros((batch_size, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        #B, HW, C

        for ihr in self.ihr_layers:
            output, ref_points = ihr(img_feat, bev_queries, bev_pos, ref_3d, self.pc_range, lidar2img, 
            self.bev_h, self.bev_w, None, img_metas, **kwargs)
            bev_queries = output 
            ref_3d = ref_points
        output = output.permute(0, 2, 1).contiguous().view(batch_size, -1, self.bev_h, self.bev_w)

        return output 
    
@NECKS.register_module()
class HeightDepthTransform(DepthLSSTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        bev_grid_map_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.only_use_height = True
        self.num_layer = 3
        self.pc_range = point_cloud_range
        ihr_layers = []
        for i in range(self.num_layer):
            in_filters = self.C
            out_filters = self.C
            ihr_layers.append(
                IHRLayer(in_filters, out_filters, self.only_use_height, num_cams=2)
            )
        self.ihr_layers = nn.ModuleList(ihr_layers)
        self.bev_h = bev_grid_map_size[0]
        self.bev_w = bev_grid_map_size[1]
        self.query_embedding = nn.Embedding(self.bev_h*self.bev_w, self.C)
        # nn.init.xavier_uniform_(self.query_embedding.weight)
        xavier_init(self.query_embedding, distribution='uniform', bias=0.)
        self.positional_encoding = LearnedPositionalEncoding(self.C // 2, self.bev_h, self.bev_w)

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

    @force_fp32()
    def get_feats_depth(self, x, d):
        B, N, C, fH, fW = x.shape # torch.Size([1, 6, 256, 32, 88])

        d = d.view(B * N, *d.shape[2:]) 
        x = x.view(B * N, C, fH, fW) # torch.Size([6, 256, 32, 88])

        d = self.dtransform(d) # torch.Size([6, 64, 32, 88])
        x = torch.cat([d, x], dim=1) # torch.Size([6, 320, 32, 88])
        x = self.depthnet(x) # [6, 198, 32, 88]

        depth = x[:, : self.D].softmax(dim=1) #通过soft max获得概率最大的深度 [6, 118, 32, 88]
        x =  x[:, self.D : (self.D + self.C)] # camera feature * depth
        return x, depth

    @force_fp32()
    def forward(
        self, points, img_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics, **kwargs
    ):
        camera2lidar = torch.inverse(lidar2camera)
        rots = camera2lidar[..., :3, :3]
        trans = camera2lidar[..., :3, 3]
        
        intrins = camera_intrinsics[..., :3, :3]
        # post_rots = img_aug_matrix[..., :3, :3]
        # post_trans = img_aug_matrix[..., :3, 3]

        # extra_rots = lidar_aug_matrix[..., :3, :3]
        # extra_trans = lidar_aug_matrix[..., :3, 3]

        batch_size = len(points)
        num_cam = len(img_metas[0]['img_shape'])
        depth = torch.zeros(batch_size, num_cam, 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            # cur_coords = points[b][:, :3].transpose(1, 0)
            cur_coords = apply_3d_transformation(points[b][:, :3].view(-1, 3), 'LIDAR', img_metas[b], reverse=True)
            cur_coords = cur_coords.transpose(1, 0)
            # cur_img_aug_matrix = img_aug_matrix[b]
            # cur_lidar_aug_matrix = lidar_aug_matrix[b] # not used?
            cur_lidar2image = lidar2img[b].float()

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) #投影到6个相机上
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) # ?
            # get 2d coords
            dist = cur_coords[:, 2, :] # 相机坐标系下的深度
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # 这都是投影到图像上的点了

            # imgaug
            # cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords) # todo?
            # cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(num_cam):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        depth_resize_list = []
        for i in range(batch_size):
            out = self.resize_feature(self.feature_size[0]*4, self.feature_size[1]*4, depth[i])
            depth_resize_list.append(out)
            
        depth = torch.stack(depth_resize_list)

        img_feat, depth_prob = self.get_feats_depth(img_feats, depth)   
        # B, N, C, fH, fW = img_feats.shape
        # img_feat = img_feats.view(B * N, C, fH, fW)

        dtype = img_feat.dtype
        ref_2d = self.get_reference_points(self.bev_h, self.bev_w, 1, 
        bs=batch_size, device=img_feat.device, dtype=dtype)
        
        ref_3d = torch.cat((ref_2d, torch.ones_like(ref_2d[..., :1])*0.55), -1)

        bev_queries = self.query_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        bev_mask = torch.zeros((batch_size, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        #B, HW, C

        for ihr in self.ihr_layers:
            output, ref_points = ihr(img_feat, bev_queries, bev_pos, ref_3d, self.pc_range, lidar2img, self.bev_h, self.bev_w, depth_prob, img_metas, **kwargs)
            bev_queries = output 
            ref_3d = ref_points
        output = output.permute(0, 2, 1).contiguous().view(batch_size, -1, self.bev_h, self.bev_w)

        return output 

def resize_feature(self, out_h, out_w, in_feat):
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
    grid = grid.unsqueeze(0)
    grid = grid.expand(in_feat.shape[0], *grid.shape[1:]).to(in_feat)
    
    out_feat = F.grid_sample(in_feat, grid=grid, mode='bilinear', align_corners=True)
    
    return out_feat

def feature_sampling(feat, reference_points, depth_prob, img_offset, pc_range, lidar_to_img, img_metas):
    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    B, D, num_query = reference_points.size()[:3]

    raw_points_shape = reference_points.shape[1:]
    new_points = []
    for i in range(B):
        x = apply_3d_transformation(reference_points[i].view(-1, 3), 'LIDAR', img_metas[i], reverse=True)
        x = x.view(raw_points_shape)
        new_points.append(x)
    reference_points = torch.stack(new_points)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

    # print('height',reference_points[..., 2:3])
    # B, D, HW, 3
    reference_points = reference_points.permute(1, 0, 2, 3).contiguous()
    # print('reference_points', reference_points.size())
    # D, B, num_query = reference_points.size()[:3]
    num_cam = lidar_to_img.size(1)
    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    lidar_to_img = lidar_to_img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points = torch.matmul(lidar_to_img.to(torch.float32), 
                                        reference_points.to(torch.float32)).squeeze(-1)

    eps = 1e-5
    referenece_depth = reference_points[..., 2:3].clone()
    bev_mask = (reference_points[..., 2:3] > eps)

    reference_points_cam = reference_points[..., 0:2] / torch.maximum(
        reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3])*eps)
    
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]


    reference_points_cam = (reference_points_cam - 0.5) * 2

    bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    # D, B, N, num_query, 1
    if depth_prob is not None:
        referenece_depth /= depth_prob.size()[1]
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
    if img_offset is not None:
        img_offset = img_offset.view(B, num_query, num_cam, -1).permute(0, 2, 1, 3).unsqueeze(0)
        # img_offset = img_offset.view(B, num_query, N, -1).permute(0, 2, 1, 3).contiguous().view(B*N, num_query, 2).unsqueeze(-2)
        img_offset[..., 0] /= W
        img_offset[..., 1] /= H
        reference_points_cam = reference_points_cam.clone() + img_offset
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

    return reference_points_cam, sampled_feat, bev_mask   

class IPFLayer(IHRLayer):

    def forward(self, inputs_img, inputs_pts, inputs_rad, bev_queries, bev_pos, reference_points, point_cloud_range, 
                lidar_to_img, bev_h, bev_w, depth_prob, img_metas, **kwargs):
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
            
        img_offset = self.img_offset(bev_queries) # B, HW, 2N
        # image_shapes=img_metas[0]['img_shape'][0]

        reference_points_3d, output, bev_mask = feature_sampling(
            inputs_img, reference_points, depth_prob, img_offset, point_cloud_range, lidar_to_img, img_metas)

        output = torch.nan_to_num(output)
        bev_mask = torch.nan_to_num(bev_mask)
        attention_weights = attention_weights.sigmoid()*bev_mask
        output = output * attention_weights
        #  B,C ,Nq, N, D
        # output = output.permute(0, 2, 1, 3, 4).flatten(2).permute(0, 2, 1).unsqueeze(-1)
        # output = self.stereo_consist(output).squeeze(-1)
        output = output.squeeze(-1).sum(-1)

        output = output.permute(0, 2, 1).contiguous()
        output = self.output_proj(output)
        output = self.dropout(output) + inp_residual
        output = output.permute(0, 2, 1).contiguous()

        # B, C, HW
        B, C = output.size()[:2]
        output = output.view(B, C, bev_h, bev_w)
        if inputs_pts is not None:
            output = torch.cat((output, inputs_pts), 1)
        if inputs_rad is not None:
            output = torch.cat((output, inputs_rad), 1)
        output = self.vtransform(output)
        # output = self.seblock(output) 
        # output = self.cbam(output) 
        output = output.flatten(2).permute(0, 2, 1).contiguous()
        # output = self.output_proj(output)

        return output, reference_points

@NECKS.register_module()
class HeightDepthFusion(HeightDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        bev_grid_map_size: Tuple[int, int],
        used_sensors: None,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            point_cloud_range=point_cloud_range,
            bev_grid_map_size=bev_grid_map_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )

        self.use_LiDAR = used_sensors.get('use_lidar', False)
        self.use_Cam = used_sensors.get('use_camera', False)
        self.use_Radar = used_sensors.get('use_radar', False)

        sensor_cnt = self.use_LiDAR + self.use_Cam + self.use_Radar
        ipf_layers = []
        for i in range(self.num_layer):
            in_filters = self.C * sensor_cnt
            out_filters = self.C
            ipf_layers.append(
                IPFLayer(in_filters, out_filters, self.only_use_height, num_cams=2)
            )
        self.ihr_layers = nn.ModuleList(ipf_layers)

    @force_fp32()
    def forward(
        self, points, img_feats, pts_feats, rad_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics, **kwargs
    ):

        batch_size = len(points)
        num_cam = len(img_metas[0]['img_shape'])
        depth = torch.zeros(batch_size, num_cam, 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            # cur_coords = points[b][:, :3].transpose(1, 0)
            cur_coords = apply_3d_transformation(points[b][:, :3].view(-1, 3), 'LIDAR', img_metas[b], reverse=True)
            cur_coords = cur_coords.transpose(1, 0)
            # cur_img_aug_matrix = img_aug_matrix[b]
            # cur_lidar_aug_matrix = lidar_aug_matrix[b] # not used?
            cur_lidar2image = lidar2img[b].float()

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) #投影到6个相机上
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) # ?
            # get 2d coords
            dist = cur_coords[:, 2, :] # 相机坐标系下的深度
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # 这都是投影到图像上的点了

            # imgaug
            # cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords) # todo?
            # cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(num_cam):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        depth_resize_list = []
        for i in range(batch_size):
            out = self.resize_feature(self.feature_size[0]*4, self.feature_size[1]*4, depth[i])
            depth_resize_list.append(out)
            
        depth = torch.stack(depth_resize_list)

        img_feat, depth_prob = self.get_feats_depth(img_feats, depth)   
        # B, N, C, fH, fW = img_feats.shape
        # img_feat = img_feats.view(B * N, C, fH, fW)

        dtype = img_feat.dtype
        ref_2d = self.get_reference_points(self.bev_h, self.bev_w, 1, 
        bs=batch_size, device=img_feat.device, dtype=dtype)
        
        ref_3d = torch.cat((ref_2d, torch.ones_like(ref_2d[..., :1])*0.55), -1)

        bev_queries = self.query_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        bev_mask = torch.zeros((batch_size, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        #B, HW, C

        for ihr in self.ihr_layers:
            output, ref_points = ihr(img_feat, pts_feats, rad_feats, bev_queries, bev_pos, ref_3d, self.pc_range, lidar2img, self.bev_h, self.bev_w, depth_prob, img_metas, **kwargs)
            bev_queries = output 
            ref_3d = ref_points
        output = output.permute(0, 2, 1).contiguous().view(batch_size, -1, self.bev_h, self.bev_w)

        return output 