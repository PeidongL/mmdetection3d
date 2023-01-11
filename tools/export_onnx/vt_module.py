import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import jit, cuda
import time
import math
import cv2

def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.contiguous().view(n, -1) 
    y = y.contiguous().view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

def bilinear_sample_noloop(image, grid):
    """
    :param image: sampling source of shape [N, C, H, W]
    :param grid: integer sampling pixel coordinates of shape [N, grid_H, grid_W, 2]
    :return: sampling result of shape [N, C, grid_H, grid_W]
    """
    Nt, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    xgrid, ygrid = grid.split([1, 1], dim=-1)
    mask = ((xgrid >= 0) & (ygrid >= 0) & (xgrid < W - 1) & (ygrid < H - 1)).float()
    x0 = torch.floor(xgrid)
    x1 = x0 + 1
    y0 = torch.floor(ygrid)
    y1 = y0 + 1
    wa = ((x1 - xgrid) * (y1 - ygrid)).permute(3, 0, 1, 2)
    wb = ((x1 - xgrid) * (ygrid - y0)).permute(3, 0, 1, 2)
    wc = ((xgrid - x0) * (y1 - ygrid)).permute(3, 0, 1, 2)
    wd = ((xgrid - x0) * (ygrid - y0)).permute(3, 0, 1, 2)
    x0 = (x0 * mask).view(Nt, grid_H, grid_W).long()
    y0 = (y0 * mask).view(Nt, grid_H, grid_W).long()
    x1 = (x1 * mask).view(Nt, grid_H, grid_W).long()
    y1 = (y1 * mask).view(Nt, grid_H, grid_W).long()
    ind = torch.arange(Nt, device=image.device) #torch.linspace(0, Nt - 1, Nt, device=image.device)
    ind = ind.view(Nt, 1).expand(-1, grid_H).view(Nt, grid_H, 1).expand(-1, -1, grid_W).long()
    image = image.permute(1, 0, 2, 3)
    output_tensor = (image[:, ind, y0, x0] * wa + image[:, ind, y1, x0] * wb + image[:, ind, y0, x1] * wc + \
                 image[:, ind, y1, x1] * wd).permute(1, 0, 2, 3)
    output_tensor *= mask.permute(0, 3, 1, 2).expand(-1, C, -1, -1)
    image = image.permute(1, 0, 2, 3)
    return output_tensor

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),dim=-1).permute(2, 0,1).contiguous().unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str

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
                 only_use_height):
        super().__init__()
        
        self.only_use_height = only_use_height
        if only_use_height:
            self.pos_offset = nn.Linear(out_channels, 1)
        else:
            self.pos_offset = nn.Linear(out_channels, 3)
        self.img_offset = nn.Linear(out_channels, 2 * 2)
        self.vtransform = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True),
        )
        self.seblock = SE_Block(out_channels)
        self.cbam = CBAM_Block(kernel_size=7)
        self.stereo_consist = nn.Sequential(
            nn.Conv2d(in_channels * 2,
                out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )


    def forward(self, inputs, inputs_side, bev_queries, bev_pos, reference_points, 
                point_cloud_range, lidar_to_img, lidar_to_img_side, image_shapes, 
                image_shapes_side, bev_h, bev_w):
        #B, HW, 3
        bev_queries = bev_queries + bev_pos
        pos_offset = self.pos_offset(bev_queries)
        if self.only_use_height:
            # reference_points = torch.cat((reference_points, pos_offset.sigmoid().unsqueeze(1)), -1)
            reference_points[...,2:3] += pos_offset.unsqueeze(1)
            reference_points[...,2:3] = reference_points[...,2:3].sigmoid()
        else:
            reference_points += pos_offset
            
        # pos_offset = pos_offset.sigmoid()
        # voxel_size[0]
        # reference_points += pos_offset
        img_offset = self.img_offset(bev_queries) # B, HW, 2N
        reference_points_3d, output, bev_mask = feature_sampling(
            inputs, reference_points, img_offset, point_cloud_range, lidar_to_img, image_shapes)
        output = torch.where(torch.isnan(output), torch.tensor(0.).cuda(), output)
        # bev_mask = torch.where(torch.isnan(bev_mask), torch.tensor(0.).cuda(), bev_mask)
        # output[torch.isnan(output)] = 0
        # bev_mask[torch.isnan(bev_mask)] = 0
        output = output * bev_mask
        # # B,C ,Nq, N, D
        output = output.permute(0, 2, 1, 3, 4).flatten(2).permute(0, 2, 1).unsqueeze(-1)
        output = self.stereo_consist(output).unsqueeze(-1)

        reference_points_3d, output_side, bev_mask_side = feature_sampling(
            inputs_side, reference_points, None, point_cloud_range, lidar_to_img_side, image_shapes_side)
        output_side = torch.where(torch.isnan(output_side), torch.tensor(0.).cuda(), output_side)
        # bev_mask = torch.where(torch.isnan(bev_mask), torch.tensor(0.).cuda(), bev_mask)
        # output_side[torch.isnan(output_side)] = 0
        # bev_mask_side[torch.isnan(bev_mask_side)] = 0
        output_side = output_side * bev_mask_side

        output = torch.cat((output, output_side), -2)

        output = output.squeeze(-1).sum(-1)
        # B, C, HW, 1
        B, C = output.size()[:2]
        output = output.view(B, C, bev_h, bev_w)
        output = self.vtransform(output)
        output = self.seblock(output) 
        output = self.cbam(output) 
        output = output.flatten(2).permute(0, 2, 1).contiguous()

        return output, reference_points

class HeightTrans(nn.Module):
    def __init__(self, model_cfg, grid_size, img_shapes, img_side_shapes, point_cloud_range):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_img_features = self.model_cfg.NUM_IMG_FEATURES
        self.only_use_height = self.model_cfg.ONLY_USE_HEIGHT
        self.num_layer = 3
        self.img_shapes = img_shapes
        self.img_side_shapes = img_side_shapes
        self.pc_range = point_cloud_range
        self.embed_dims = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        self.query_embedding = nn.Embedding(self.ny*self.nx,
                                                self.embed_dims)
        # nn.init.xavier_uniform_(self.query_embedding.weight)
        self.positional_encoding = LearnedPositionalEncoding(self.embed_dims // 2, self.ny, self.nx)

        ihr_layers = []
        for i in range(self.num_layer):
            in_filters = self.num_img_features
            out_filters = self.embed_dims
            ihr_layers.append(
                IHRLayer(in_filters, out_filters, self.only_use_height)
            )
        self.ihr_layers = nn.ModuleList(ihr_layers)

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
        # ref_y, ref_x = torch.meshgrid(
        #     torch.linspace(
        #         0.5, H - 0.5, H, dtype=dtype, device=device),
        #     torch.linspace(
        #         0.5, W - 0.5, W, dtype=dtype, device=device)
        # )
        ref_y, ref_x = torch.meshgrid(
            torch.arange(
                0.5, H + 0.5, 1, dtype=dtype, device=device),
            torch.arange(
                0.5, W + 0.5, 1, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(1)
        return ref_2d

    def forward(self, front_left_features, front_right_features,
                side_left_features, side_right_features,
                lidar_to_img_front_left, lidar_to_img_front_right,
                lidar_to_img_side_left, lidar_to_img_side_right):
        
        front_features = torch.cat((front_left_features.permute((0, 3, 1, 2)), front_right_features.permute((0, 3, 1, 2))), dim=0)
        side_features = torch.cat((side_left_features.permute((0, 3, 1, 2)), side_right_features.permute((0, 3, 1, 2))), dim=0)
        lidar_to_img = torch.cat((lidar_to_img_front_left, lidar_to_img_front_right), dim=1)
        lidar_to_img_side = torch.cat((lidar_to_img_side_left, lidar_to_img_side_right), dim=1)

        dtype = front_features[0].dtype

        ref_2d = self.get_reference_points(self.ny, self.nx, self.nz, 
        bs=1, device=front_features[0].device, dtype=dtype)
        
        ref_3d = torch.cat((ref_2d, torch.zeros_like(ref_2d[..., :1])), -1)

        bev_queries = self.query_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(1, 1, 1)
        bev_mask = torch.zeros((1, self.ny, self.nx),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()
        #B, HW, C
        for ihr in self.ihr_layers:
            output, ref_points = ihr(front_features, side_features, 
                bev_queries, bev_pos, ref_3d, self.pc_range, lidar_to_img, lidar_to_img_side, 
                self.img_shapes, self.img_side_shapes, self.ny, self.nx)
            bev_queries = output 
            ref_3d = ref_points
        output = output.permute(0, 2, 1).contiguous().view(1, -1, self.ny, self.nx)
        # #B, C , H ,W
        return output

def feature_sampling(feat, reference_points, img_offset, pc_range, lidar_to_img, image_shapes):
    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # print(reference_points[..., :3])
    # reference_points (B, D, num_queries, 4)

    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    # print('reference_points', reference_points[...,0:3])
    # B, D, HW, 3
    reference_points = reference_points.permute(1, 0, 2, 3).contiguous()
    # print('reference_points', reference_points.size())
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar_to_img.size(1)
    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    lidar_to_img = lidar_to_img.view(
        1, B, num_cam, 1, 3, 4).repeat(D, 1, 1, num_query, 1, 1)
    # image_shapes = image_shapes.view(
    #     1, B, num_cam, 1, 2).repeat(D, 1, 1, num_query, 1)

    reference_points_cam = torch.matmul(lidar_to_img.to(torch.float32), 
                                        reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    bev_mask = (reference_points_cam[..., 2:3] > eps)
    depth = reference_points_cam[..., 2:3].clone()
    # depth[depth < eps] = eps
    # print('reference_points_cam', reference_points_cam.size())
    # print('bev_mask', bev_mask.size())
    # reference_points_cam[..., 2:3][reference_points_cam[..., 2:3] < eps] = eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.where(depth>eps, depth, torch.ones_like(depth)*eps)
    # print(image_shapes[..., 1])
    reference_points_cam[..., 0] /= image_shapes[1]
    reference_points_cam[..., 1] /= image_shapes[0]
    # reference_points_cam[..., 1] = (reference_points_cam[..., 1] - image_shapes[..., 2]) / image_shapes[..., 0]
    # print(reference_points_cam[..., 0].size())
    # print(image_shapes[..., 1].size())
    reference_points_cam = (reference_points_cam - 0.5) * 2
    bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))

    # bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > 0.0) 
    #              & (reference_points_cam[..., 0:1] < 1.0) 
    #              & (reference_points_cam[..., 1:2] > 0.0) 
    #              & (reference_points_cam[..., 1:2] < 1.0))
    # D, B, N, num_query, 1

    bev_mask = bev_mask.permute(1, 4, 3, 2, 0).contiguous()
    # B,1 ,Nq, N, D

    # for lvl, feat in enumerate(mlvl_feats):[]
    N, C, H, W = feat.size()
    # feat = feat.view(B*N, C, H, W)
    reference_points_cam_lvl = reference_points_cam.permute(1, 2, 3, 0, 4).contiguous().view(N, num_query, D, 2).to(torch.float32)
    if img_offset is not None:
        img_offset = img_offset.view(1, num_query, N, -1).permute(0, 2, 1, 3).contiguous().view(N, num_query, 2).unsqueeze(-2)
        img_offset[..., 0] /= W
        img_offset[..., 1] /= H
        reference_points_cam_lvl = reference_points_cam_lvl.clone() + img_offset
    sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
    sampled_feat = sampled_feat.view(1, N, C, num_query, D).permute(0, 2, 3, 1, 4).contiguous()
    # B,C ,Nq, N, D

    return reference_points_cam, sampled_feat, bev_mask
