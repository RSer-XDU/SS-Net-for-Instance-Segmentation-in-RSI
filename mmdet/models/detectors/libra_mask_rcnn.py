import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init


from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

from ..plugins import NonLocal2D
from ..utils import ConvModule

@DETECTORS.register_module
class Libra_Mask_RCNN(BaseDetector, RPNTestMixin, BBoxTestMixin,
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
                 ):
        super(Libra_Mask_RCNN, self).__init__()

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
        super(Libra_Mask_RCNN, self).init_weights(pretrained)
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


            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)


        return losses

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

        # step 2: refine gathered features
        if self.semantic_fpn_refine_type is not None:
            bsf = self.refine_0(bsf)



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
            segm_results = self.simple_test_mask(
                outs, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
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
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
