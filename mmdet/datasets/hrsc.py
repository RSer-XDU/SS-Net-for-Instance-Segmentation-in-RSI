import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset


class HRSCDataset(CocoDataset):

    CLASSES = ("ship",)



