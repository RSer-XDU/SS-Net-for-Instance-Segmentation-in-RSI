from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS

@ROI_EXTRACTORS.register_module
class PASingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(PASingleRoIExtractor, self).__init__()
        cfg = roi_layer.copy()
        self.roi_size = cfg.pop('out_size')
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.hidden_dim = 1024
        self.featmap_strides = featmap_strides
        self.num_levels = len(featmap_strides)
        self.finest_scale = finest_scale


    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = torch.cuda.FloatTensor(rois.size()[0], self.out_channels,
                                           out_size, out_size).fill_(0)
        roi_feats_all_level = []
        for i in range(num_levels):
            # every rois use every FPN features
            roi_feats_all_level.append(self.roi_layers[i](feats[i], rois))
        for i in range(len(roi_feats_all_level)):
            roi_feats_all_level[0] = torch.max(roi_feats_all_level[0], roi_feats_all_level[i])
        # print(roi_feats_all_level[0])
        roi_feats = roi_feats_all_level[0]

        return roi_feats
