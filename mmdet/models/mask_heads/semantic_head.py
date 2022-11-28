import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init

from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class SemanticHead(nn.Module):
    """single-level fused semantic segmentation head.

                                            /-> 1x1 conv (mask prediction)
    in_4 -> 1x1 conv -----> 3x3 convs (*4)
                                            \-> 1x1 conv (feature)
 
    """  # noqa: W605

    def __init__(self,
                 fusion_level,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=0.2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticHead, self).__init__()

        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg


        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_embedding = ConvModule(
            conv_out_channels,
            conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def init_weights(self):
        kaiming_init(self.conv_logits)

    def forward(self, x):

        for i in range(self.num_convs):
            x = self.convs[i](x)
        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        return mask_pred, x

    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg
