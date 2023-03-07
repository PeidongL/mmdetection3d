mport copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32

@HEADS.register_module()
class BEVHeightOccHead(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 use_3d=False,
                 use_conv=False,
                 num_classes=18,
                 out_dim=32,
                 pillar_h=16,
                 loss_occ=None,
                 use_mask=False,
                 act_cfg=dict(type='ReLU',inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):
        super(TransformerOcc, self).__init__(**kwargs)

        self.embed_dims = embed_dims
        self.fp16_enabled = False
        self.use_mask=use_mask
        self.loss_occ = build_loss(loss_occ)

        self.use_3d=use_3d
        self.use_conv=use_conv
        self.num_classes=num_classes
        self.pillar_h = pillar_h
        self.out_dim=out_dim
        if not use_3d:
            if use_conv:
                use_bias = norm_cfg is None
                self.decoder  = nn.Sequential(
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims*2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),)

            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims * 2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims * 2, self.embed_dims*2),
                )
        else:
            use_bias_3d = norm_cfg_3d is None

            self.middle_dims=self.embed_dims//pillar_h
            self.decoder = nn.Sequential(
                ConvModule(
                    self.middle_dims,
                    self.out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
                ConvModule(
                    self.out_dim,
                    self.out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
            )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2,num_classes),
        )

    def forward(self, bev_embed, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        if self.use_3d:
            outputs=self.decoder(bev_embed.view(bs,-1,self.pillar_h,bev_h, bev_w))
            outputs=outputs.permute(0,4,3,2,1)

        elif self.use_conv:

            outputs = self.decoder(bev_embed)
            outputs = outputs.view(bs, -1,self.pillar_h, bev_h, bev_w).permute(0,3,4,2, 1)
        else:
            outputs = self.decoder(bev_embed.permute(0,2,3,1))
            outputs = outputs.view(bs, bev_h, bev_w,self.pillar_h,self.out_dim)
        outputs = self.predicter(outputs)

        outs = {
            'bev_embed': bev_embed,
            'occ':outputs,
        }

        return outs

@force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()
        occ=preds_dicts['occ']
        assert voxel_semantics.min()>=0 and voxel_semantics.max()<=17
        losses = self.loss_single(voxel_semantics,mask_camera,occ)
        loss_dict['loss_occ']=losses
        return loss_dict

    def loss_single(self,voxel_semantics,mask_camera,preds):
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)


        return occ_score
