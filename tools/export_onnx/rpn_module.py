# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn as nn
import torch
import numpy as np


class Anchor3DHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 use_direction_classifier=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels

        self.use_direction_classifier = use_direction_classifier
        self.num_anchors = num_classes * 2
        self.box_code_size = 7

        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def forward(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

class PcdetBackbone(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 num_filters=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[128, 128, 128]):
        super(PcdetBackbone, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(num_upsample_filters)

        num_levels = len(layer_nums)
        c_in_list = [in_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in


    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        return x

class RPN(nn.Module):
    def __init__(self, backbone_cfg, head_cfg):
        super().__init__()
        self.backbone = PcdetBackbone(in_channels=backbone_cfg.in_channels,
                                      layer_nums=backbone_cfg.layer_nums,
                                      layer_strides=backbone_cfg.layer_strides,
                                      num_filters=backbone_cfg.num_filters,
                                      upsample_strides=backbone_cfg.upsample_strides,
                                      num_upsample_filters=backbone_cfg.num_upsample_filters)
        
        self.bbox_head = Anchor3DHead(num_classes=head_cfg.num_classes,
                                 in_channels=head_cfg.in_channels,
                                 feat_channels=head_cfg.feat_channels,
                                 use_direction_classifier=head_cfg.use_direction_classifier,
                                 )
        
    def forward(self, input):
        x = self.backbone(input) # return [out]
        out = self.bbox_head(x) # return cls_score, bbox_pred, dir_cls_preds
        cls_score, bbox_pred, dir_cls_preds = out
        cls_score = cls_score
        bbox_pred = bbox_pred
        dir_cls_preds = dir_cls_preds
        
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        return cls_score, bbox_pred, dir_cls_preds