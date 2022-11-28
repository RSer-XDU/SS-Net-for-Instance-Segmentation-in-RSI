import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .recall import eval_recalls


def coco_eval(result_file, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_file, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    assert result_file.endswith('.json')
    coco_dets = coco.loadRes(result_file)

    img_ids = coco.getImgIds()
    for res_type in result_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['category_id'] = dataset.cat_ids[label]
                data['score'] = float(bboxes[i][4])
                json_results.append(data)
    return json_results


    # category_instancesonly = [
    #     'unlabeled',
    #     'ship',  1  5
    #     'storage_tank', 2   3 
    #     'baseball_diamond', 3   11
    #     'tennis_court', 4   8
    #     'basketball_court', 5   13
    #     'Ground_Track_Field', 6 9
    #     'Bridge', 7 12
    #     'Large_Vehicle', 8  1
    #     'Small_Vehicle', 9   2
    #     'Helicopter', 10    15
    #     'Swimming_pool', 11 6
    #     'Roundabout', 12    14
    #     'Soccer_ball_field', 13 10
    #     'plane', 14 4
    #     'Harbor' 15 7
    # ]




def segm2json(dataset, results):
 
    CLASS_NUM_INDEX = [8, 9, 2, 14, 1, 11, 15, 4, 6, 13, 3, 7, 5, 12, 10]
    print(dataset.cat_ids)
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])

                # data['category_id'] = CLASS_NUM_INDEX[label]
                data['category_id'] = dataset.cat_ids[label] # for NWPU
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                data['score'] = float(bboxes[i][4])
                json_results.append(data)
    return json_results


def results2json(dataset, results, out_file):
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
    else:
        raise TypeError('invalid type of results')
    mmcv.dump(json_results, out_file)
