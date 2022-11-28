import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import ms_mask_target


@HEADS.register_module
class MSPAFCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(MSPAFCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mask = build_loss(loss_mask)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convfc4 = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                self.conv_kernel_size,
                padding=padding,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        self.convfc5 = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels // 2 ,
                self.conv_kernel_size,
                padding=padding,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        self.mask_fc =nn.Sequential(
                nn.Linear( int(self.conv_out_channels // 2) * (self.roi_feat_size) ** 2,  (2 * self.roi_feat_size) ** 2, bias=True),
                nn.ReLU(inplace=True))

        self.avgpool = nn.AvgPool2d(2,2)
        self.mask_upsampling = nn.Upsample(
                scale_factor=2, mode='bilinear')

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)

        self.conv_logits1 = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits2 = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits3 = nn.Conv2d(logits_in_channel, out_channels, 1)

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits1, self.conv_logits2, self.conv_logits3]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        # nn.init.xavier_uniform_(self.mask_fc.weight)
        # nn.init.constant_(self.mask_fc.bias, 0)


    def forward(self, x):
        input_feature = x

        for conv in self.convs:
            x = conv(x)

        s1_mask_pred = self.conv_logits2(x)
       # mask_pred1  = self.mask_upsampling(s1_mask_pred)

        downsampling_x = self.avgpool(x)
        s2_mask_pred = self.conv_logits1(downsampling_x)
        #mask_pred2 = self.mask_upsampling(s2_mask_pred)
        #mask_pred2 = self.mask_upsampling(mask_pred2)

        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)

        for inter in range(3):
            input_feature =  self.convs[inter](input_feature)

        input_feature = self.convfc4(input_feature)
        input_feature = self.convfc5(input_feature)
        input_feature = input_feature.reshape(input_feature.shape[0],-1)
        input_feature = self.mask_fc(input_feature)
        input_feature = input_feature.reshape(x.shape[0],1, x.shape[2], x.shape[3])
        if not self.class_agnostic:
            input_feature = input_feature.repeat(1, self.num_classes, 1, 1)
        mask_pred = self.conv_logits3(x)
        s0_mask_pred = mask_pred + input_feature 

        return s0_mask_pred, s1_mask_pred, s2_mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        s0_mask_targets, s1_mask_targets, s2_mask_targets = ms_mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return s0_mask_targets, s1_mask_targets, s2_mask_targets


    def loss(self, s0_mask_pred, s1_mask_pred, s2_mask_pred, s0_mask_targets, s1_mask_targets, s2_mask_targets, labels):
        loss = dict()
        assert s0_mask_pred.size(0) == s1_mask_pred.size(0) == s2_mask_pred.size(0)

        if self.class_agnostic:
            s0_loss_mask = self.loss_mask(s0_mask_pred, s0_mask_targets,
                                       torch.zeros_like(labels))
            s1_loss_mask = self.loss_mask(s1_mask_pred, s1_mask_targets,
                                       torch.zeros_like(labels))
            s2_loss_mask = self.loss_mask(s2_mask_pred, s2_mask_targets,
                                       torch.zeros_like(labels))
        else:
            s0_loss_mask = self.loss_mask(s0_mask_pred, s0_mask_targets, labels)
            s1_loss_mask = self.loss_mask(s1_mask_pred, s1_mask_targets, labels)
            s2_loss_mask = self.loss_mask(s2_mask_pred, s2_mask_targets, labels)

        loss['s0_loss_mask'] = s0_loss_mask
        loss['s1_loss_mask'] = s1_loss_mask
        loss['s2_loss_mask'] = s2_loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
            
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)


        cls_segms = [[] for _ in range(self.num_classes - 1)]
        mask_scores = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bbox_mask = mmcv.imresize(mask_pred_, (w, h))

            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)


        return cls_segms


    def get_combine_seg_masks(self, s0_mask_pred, s1_mask_pred, s2_mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
            
        """
        if isinstance(s0_mask_pred, torch.Tensor):
            s0_mask_pred = s0_mask_pred.sigmoid().cpu().numpy()

        if isinstance(s1_mask_pred, torch.Tensor):
            s1_mask_pred = s1_mask_pred.sigmoid().cpu().numpy()
        
        if isinstance(s2_mask_pred, torch.Tensor):
            s2_mask_pred = s2_mask_pred.sigmoid().cpu().numpy()        
        
        assert isinstance(s0_mask_pred, np.ndarray) and isinstance(s1_mask_pred, np.ndarray) and isinstance(s2_mask_pred, np.ndarray)


        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                s0_mask_pred_ = s0_mask_pred[i, label, :, :]
                s1_mask_pred_ = s1_mask_pred[i, label, :, :]
                s2_mask_pred_ = s2_mask_pred[i, label, :, :]
            else:
                s0_mask_pred_ = s0_mask_pred[i, 0, :, :]
                s1_mask_pred_ = s1_mask_pred[i, 0, :, :]
                s2_mask_pred_ = s2_mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox0_mask = mmcv.imresize(s0_mask_pred_, (w, h))
            bbox1_mask = mmcv.imresize(s1_mask_pred_, (w, h))
            bbox2_mask = mmcv.imresize(s2_mask_pred_, (w, h))
            
            bbox0_mask = (bbox0_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            bbox1_mask = (bbox1_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            bbox2_mask = (bbox2_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)

            bbox_mask = bbox0_mask + bbox1_mask + bbox2_mask
            bbox_mask = (bbox_mask > 1).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)


        return cls_segms