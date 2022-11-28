from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map


def voc_eval(det_results, config, iou_thr=0.5):
    # cfg = mmcv.Config.fromfile(config)
    test_dataset = mmcv.runner.obj_from_dict(config.data.test, datasets)
    # det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(test_dataset)):
        ann = test_dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = gt_ignore
    if hasattr(test_dataset, 'year') and test_dataset.year == 2007:
        test_dataset_name = 'voc07'
    else:
        test_dataset_name = test_dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=test_dataset_name,
        print_summary=True)

