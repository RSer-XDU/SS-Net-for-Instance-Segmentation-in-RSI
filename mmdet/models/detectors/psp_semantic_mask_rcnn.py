import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import mmcv


from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, bbox_mapping, merge_aug_masks

from ..plugins import NonLocal2D
from ..utils import ConvModule

@DETECTORS.register_module
class PSP_Semantic_Mask_RCNN(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 semantic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 semantic_fpn_in_channels = 256,
                 semantic_fpn_num_levels = 5,
                 semantic_fpn_refine_level=2,
                 semantic_fpn_refine_type=None,
                 semantic_fpn_conv_cfg=None,
                 semantic_fpn_norm_cfg=None,
                 semantic_fpn_method = "multiply",
                 ):
        super(PSP_Semantic_Mask_RCNN, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        
        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        if semantic_head is not None:
            self.semantic_head = builder.build_head(semantic_head)
        #semantic + balance fpn
        self.semantic_fpn_method = semantic_fpn_method

        assert semantic_fpn_refine_type in [None, 'conv', 'non_local']
        self.semantic_fpn_in_channels = semantic_fpn_in_channels
        self.semantic_fpn_num_levels = semantic_fpn_num_levels
        self.semantic_fpn_refine_level = semantic_fpn_refine_level
        self.semantic_fpn_refine_type = semantic_fpn_refine_type

        self.semantic_fpn_conv_cfg = semantic_fpn_conv_cfg
        self.semantic_fpn_norm_cfg = semantic_fpn_norm_cfg

        assert 0 <= self.semantic_fpn_refine_level < self.semantic_fpn_num_levels

        if self.semantic_fpn_refine_type == 'conv':
            self.refine_0 = ConvModule(
                self.semantic_fpn_in_channels,
                self.semantic_fpn_in_channels,
                3,
                padding=1,
                conv_cfg=self.semantic_fpn_conv_cfg,
                norm_cfg=self.semantic_fpn_norm_cfg)
        elif self.semantic_fpn_refine_type == 'non_local':
            self.refine_0 = NonLocal2D(
                self.semantic_fpn_in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.semantic_fpn_conv_cfg,
                norm_cfg=self.semantic_fpn_norm_cfg)
        self.refine_1_0 = ConvModule(
                2 * self.semantic_fpn_in_channels,
                self.semantic_fpn_in_channels,
                3,
                padding=1,
                conv_cfg=self.semantic_fpn_conv_cfg,
                norm_cfg=self.semantic_fpn_norm_cfg)
        self.refine_1_1 = ConvModule(
                self.semantic_fpn_in_channels,
                self.semantic_fpn_in_channels,
                1,
                padding=0,
                conv_cfg=self.semantic_fpn_conv_cfg,
                norm_cfg=self.semantic_fpn_norm_cfg)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def init_weights(self, pretrained=None):
        super(PSP_Semantic_Mask_RCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x= self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)

        feats = []
        gather_size = x[self.semantic_fpn_refine_level].size()[2:]
        for i in range(self.semantic_fpn_num_levels):
            if i < self.semantic_fpn_refine_level:
                gathered = F.adaptive_max_pool2d(
                    x[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    x[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.semantic_fpn_refine_type is not None:
            bsf = self.refine_0(bsf)

        # semantic segmentation part
        #  segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(bsf)


            if self.semantic_fpn_method == 'multiply':
                bsf = bsf * semantic_feat
            elif self.semantic_fpn_method == 'concate':
                bsf = torch.cat([bsf , semantic_feat], dim=1)
                bsf = self.refine_1_0(bsf)
                bsf = self.refine_1_1(bsf)
            else:
                raise ValueError('just multiply and concate')



        # step 3: scatter refined features to multi-levels by a residual path
        x_outs = []
        for i in range(self.semantic_fpn_num_levels):
            out_size = x[i].size()[2:]
            if i < self.semantic_fpn_refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            x_outs.append(residual + x[i])


        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x_outs)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x_outs[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)


            refine_mask_pred, s0_mask_pred, s1_mask_pred, s2_mask_pred = self.mask_head(mask_feats)
            outs = outs + (refine_mask_pred, s0_mask_pred, s1_mask_pred, s2_mask_pred , )


        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None):


        losses = dict()

        x = self.extract_feat(img)

        # semantic and banlance fpn
        assert len(x) == self.semantic_fpn_num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = x[self.semantic_fpn_refine_level].size()[2:]
        for i in range(self.semantic_fpn_num_levels):
            if i < self.semantic_fpn_refine_level:
                gathered = F.adaptive_max_pool2d(
                    x[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    x[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.semantic_fpn_refine_type is not None:
            bsf = self.refine_0(bsf)

        # # semantic segmentation part
        # #  segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(bsf)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg

            if self.semantic_fpn_method == 'multiply':
                bsf = bsf * semantic_feat
            elif self.semantic_fpn_method == 'concate':
                bsf = torch.cat([bsf , semantic_feat], dim=1)
                bsf = self.refine_1_0(bsf)
                bsf = self.refine_1_1(bsf)
            else:
                raise ValueError('just multiply and concate')

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.semantic_fpn_num_levels):
            out_size = x[i].size()[2:]
            if i < self.semantic_fpn_refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + x[i])



        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(outs)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)


            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in outs])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                outs[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)


            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
        
        
        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    outs[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]


            refine_mask_pred, s0_mask_pred, s1_mask_pred, s2_mask_pred = self.mask_head(mask_feats)

            s0_mask_targets, s1_mask_targets, s2_mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(refine_mask_pred, s0_mask_pred, s1_mask_pred, s2_mask_pred, s0_mask_targets, s1_mask_targets, s2_mask_targets,
                                            pos_labels)
            losses.update(loss_mask)


        return losses

    def _mask_forward_test(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            refine_mask_pred, _, _, _ = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                refine_mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."


        x = self.extract_feat(img)

        assert len(x) == self.semantic_fpn_num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = x[self.semantic_fpn_refine_level].size()[2:]
        for i in range(self.semantic_fpn_num_levels):
            if i < self.semantic_fpn_refine_level:
                gathered = F.adaptive_max_pool2d(
                    x[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    x[i], size=gather_size, mode='nearest')
            feats.append(gathered)
 
        bsf = sum(feats) / len(feats)
   
        # import matplotlib.pyplot as plt
        # bsf_show = torch.mean(F.relu(bsf), dim=1)
        # bsf_show = torch.squeeze(bsf_show)
        # bsf_show = bsf_show.data.cpu().numpy()
        # import numpy as np

        # plt.subplot(121)
        # plt.imshow(bsf_show, cmap='jet')
        # plt.show()

        # step 2: refine gathered features
        if self.semantic_fpn_refine_type is not None:
            bsf = self.refine_0(bsf)


        # bsf_show = torch.mean(bsf, dim=1)
        # bsf_show = torch.squeeze(bsf_show)
        # bsf_show = bsf_show.data.cpu().numpy()
        # import numpy as np
        # print(np.min(bsf_show))
        # # bsf_show = (bsf_show - np.min(bsf_show)) / (np.max(bsf_show) - np.min(bsf_show))
        # plt.subplot(142)
        # plt.imshow(bsf_show, cmap='jet')
        # plt.show()

        # semantic segmentation part
        #  segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(bsf)
            bsf_show = torch.mean(bsf, dim = 1)
            bsf_show = torch.squeeze(bsf_show)
            
            if self.semantic_fpn_method == 'multiply':
                bsf = bsf * semantic_feat
            elif self.semantic_fpn_method == 'concate':
                bsf = torch.cat([bsf , semantic_feat], dim=1)
                bsf = self.refine_1_0(bsf)
                bsf = self.refine_1_1(bsf)
            else:
                raise ValueError('just multiply and concate')

        # semantic_pred2,_ = torch.max(torch.squeeze(F.softmax(semantic_pred,dim=1))[1:],dim=0)

        # print(semantic_pred2.size())
        # bsf_show = bsf * torch.exp(semantic_pred2)
        # bsf_show = torch.mean(F.relu(bsf_show), dim = 1)
        # bsf_show = torch.squeeze(bsf_show)
        # bsf_show = bsf_show.data.cpu().numpy()
        # # bsf_show = (bsf_show - np.min(bsf_show)) / (np.max(bsf_show) - np.min(bsf_show))
        # # semantic_pred2 = semantic_pred2.cpu().numpy()  
        # # semantic_pred_show = mmcv.imresize(semantic_pred2, (800, 800))     
        # plt.subplot(122)
        # plt.imshow(bsf_show,cmap='jet')
        # plt.show()
        # # print(semantic_pred2)
        # # print(torch.exp(semantic_pred2))
        # # bsf = bsf * torch.exp(semantic_pred2)


        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.semantic_fpn_num_levels):
            out_size = x[i].size()[2:]
            if i < self.semantic_fpn_refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + x[i])


        proposal_list = self.simple_test_rpn(
            outs, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            outs, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self._mask_forward_test(
                outs, img_meta, det_bboxes, det_labels,  rescale=rescale)
            return bbox_results, segm_results

    def _aug_mask_forward_test(self,
                         feats,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)


        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                refine_mask_pred, _, _, _ = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(refine_mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result





    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        assert self.with_bbox, "Bbox head must be implemented."
        features_list = []
        for img in imgs:

            x = self.extract_feat(img)

            assert len(x) == self.semantic_fpn_num_levels

            # step 1: gather multi-level features by resize and average
            feats = []
            gather_size = x[self.semantic_fpn_refine_level].size()[2:]
            for i in range(self.semantic_fpn_num_levels):
                if i < self.semantic_fpn_refine_level:
                    gathered = F.adaptive_max_pool2d(
                        x[i], output_size=gather_size)
                else:
                    gathered = F.interpolate(
                        x[i], size=gather_size, mode='nearest')
                feats.append(gathered)
    
            bsf = sum(feats) / len(feats)
    

            # step 2: refine gathered features
            if self.semantic_fpn_refine_type is not None:
                bsf = self.refine_0(bsf)


            # semantic segmentation part
            #  segmentation prediction and embedded features
            if self.with_semantic:
                semantic_pred, semantic_feat = self.semantic_head(bsf)
                bsf_show = torch.mean(bsf, dim = 1)
                bsf_show = torch.squeeze(bsf_show)
                
                if self.semantic_fpn_method == 'multiply':
                    bsf = bsf * semantic_feat
                elif self.semantic_fpn_method == 'concate':
                    bsf = torch.cat([bsf , semantic_feat], dim=1)
                    bsf = self.refine_1_0(bsf)
                    bsf = self.refine_1_1(bsf)
                else:
                    raise ValueError('just multiply and concate')


            # step 3: scatter refined features to multi-levels by a residual path
            outs = []
            for i in range(self.semantic_fpn_num_levels):
                out_size = x[i].size()[2:]
                if i < self.semantic_fpn_refine_level:
                    residual = F.interpolate(bsf, size=out_size, mode='nearest')
                else:
                    residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
                outs.append(residual + x[i])
            
            features_list.append(outs)

            


        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            features_list, img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            features_list, img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self._aug_mask_forward_test(
                features_list, img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
