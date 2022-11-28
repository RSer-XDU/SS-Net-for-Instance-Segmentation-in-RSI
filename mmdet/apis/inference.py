import warnings
import os
import cv2
import matplotlib.pyplot as plt

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector




#coding=utf-8
#############################################
# mask nms　实现
#############################################
import cv2
import numpy as np
import imutils
import copy

import matplotlib.pyplot as plt


def mask_nms(dets, mask, thres=0.1):
    """
    mask nms 实现函数
    :param dets: 检测结果，是一个N*9的numpy,
    :param mask: 当前检测的mask
    :param thres: 检测的阈值
    """
    # 获取bbox及对应的score
    bbox_infos=dets[:,:4]
    scores=dets[:,4]
    # print(bbox_infos)
    # print(scores)

    keep=[]
    order=scores.argsort()[::-1]
    # print("order:{}".format(order))
    nums=len(bbox_infos)
    suppressed=np.zeros((nums), dtype=np.int)
    # print("lens:{}".format(nums))

    # 循环遍历
    for i in range(nums):
        idx=order[i]
        if suppressed[idx]==1:
            continue
        keep.append(idx)

        mask_a = mask[idx]
        area_a=np.sum(mask[idx])

        for j in range(i,nums):
            idx_j=order[j]
            if suppressed[idx_j]==1:
                continue

            mask_b = mask[idx_j]
            area_b = np.sum(mask[idx_j])
            overlap = np.sum(mask_a * mask_b)
            iou = overlap / (area_a + area_b - overlap + 1e-7)        
            # print("area_a:{},area_b:{},inte:{},mmi:{}".format(area_a,area_b,area_intersect,mmi))
            if iou >= thres:
                suppressed[idx_j] = 1
    return keep







def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def inference_center(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_center(model, imgs, img_transform, device)





def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def _inference_center(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model.vis_center(**data)
    return result, data


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)





def show_result_with_mask_nms(img, result, class_names, score_thr=0.1, out_file=None, mask_postprocess=True):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    print('score_thr', score_thr)
    img = mmcv.imread(img, flag='unchanged')
    # img = plt.imread(img)
    
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None


    if segm_result is not None:
        assert len(bbox_result) == len(segm_result)
        refine_bboxes = []
        refine_labels = []
        refine_segms = []
        before_nms = 0
        after_nms = 0
        for i in range(len(bbox_result)):
            temp_bbox = bbox_result[i]
            temp_label = np.full(temp_bbox.shape[0], i, dtype=np.int32)
            temp_mask = segm_result[i]
            inds = np.where(temp_bbox[:,-1] > score_thr)[0]
            if temp_bbox.shape[0] == 0 or len(inds) == 0:
                pass
            else:
                before_nms = before_nms + len(inds)
                masks_result = []
                bboxes_result = []
                labels_result = []
                segms_result = []
                for j in inds:
                    mask_valid = maskUtils.decode(temp_mask[j])
                    bbox_valid = temp_bbox[j,:]
                    label_valid = temp_label[j]
                    masks_result.append(mask_valid)
                    segms_result.append(temp_mask[j])
                    bboxes_result.append(bbox_valid)
                    labels_result.append(label_valid)
                masks_result = np.array(masks_result)
                segms_result = np.array(segms_result)
                bboxes_result = np.array(bboxes_result)
                labels_result = np.array(labels_result)
                keep_masks = mask_nms(bboxes_result, masks_result)
                after_nms = after_nms + len(keep_masks)
                refine_bboxes.append(bboxes_result[keep_masks])
                refine_labels.append(labels_result[keep_masks])
                refine_segms.append(segms_result[keep_masks])

    print('before_nms', before_nms)
    print('after_nms', after_nms)


    refine_bboxes = np.vstack(refine_bboxes)
    det_scores = refine_bboxes[:,-1]
    refine_labels = np.concatenate(refine_labels)


    refine_segms = mmcv.concat_list(refine_segms)

    # for refine_segm in refine_segms:
    #         color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #         refine_mask = maskUtils.decode(refine_segm).astype(np.bool)
    #         img[refine_mask] = img[refine_mask] * 0.4 + color_mask * 0.6


    assert refine_bboxes.ndim == 2
    assert refine_labels.ndim == 1
    assert refine_bboxes.shape[0] == refine_labels.shape[0]
    assert refine_bboxes.shape[1] == 4 or refine_bboxes.shape[1] == 5

    for bbox, label,score , refine_segm in zip(refine_bboxes, refine_labels, det_scores, refine_segms):
       

        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        refine_mask = maskUtils.decode(refine_segm).astype(np.bool)
        img[refine_mask] = img[refine_mask] * 0.2 + color_mask * 0.8
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        color_mask = np.squeeze(color_mask)
        # print(color_mask)
        # cv2.rectangle(
            # img, left_top, right_bottom, (int(color_mask[0]), int(color_mask[1]), int(color_mask[2])), thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(score)
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

    plt.imshow(img)
    plt.show()
    return img


def write_hbb_bbox_result(img_name, result, dataset='coco', score_thr=0.3, out_file=None):

    img_id = img_name
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    #bbox_result 返回的是每个class类别的bbox
    if bbox_result is not None:
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

        labels = np.concatenate(labels)

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    for bbox, label in zip(bboxes, labels):
        class_name = class_names[label]
        score = bbox[-1]
        f = open(os.path.join(out_file, 'Task2_' + class_name + '.txt'), 'a')
        box = bbox.astype(np.int32)
        f.write('{} {} {} {} {} {} \n'.format(img_id[:-4],
                                              float(score),
                                              int(box[0]),
                                              int(box[1]),
                                              int(box[2]),
                                              int(box[3])))


def write_obb_bbox_result(img_name, result, class_names, out_file=None, score_thr=0.01):

    img_id = img_name
    assert isinstance(class_names, (tuple, list))
    assert isinstance(result, tuple)
    bbox_result, segm_result = result

       
    #bbox_result 返回的是每个class类别的bbox
    if bbox_result is not None:
        # print('bbox', bbox_result[0].shape)
        # print(bbox_result[0])
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

        labels = np.concatenate(labels)


    bboxes_obb = []
    final_inds = []
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        # print('seg',segms)
        # print(bboxes[:,-1])
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            
            color_mask = np.random.randint(
                0, 256, (3), dtype=np.uint8).tolist()
            mask = maskUtils.decode(segms[i]).astype(np.uint8)
            import cv2
            # print(mask)
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                continue
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            bboxes_obb.append([box[0][0],box[0][1],
                                box[1][0],box[1][1],
                                box[2][0],box[2][1],
                                box[3][0],box[3][1]])
            final_inds.append(i)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[final_inds]
    labels = labels[final_inds]
    bboxes_obb = np.array(bboxes_obb) # 这儿只是坐标,得分取 bboxes 中的得分
    for bbox, bbox_obb, label in zip(bboxes, bboxes_obb, labels):
        class_name = class_names[label]
        score = bbox[-1]
        # print(bbox)
        f = open(os.path.join(out_file, 'Task1_' + class_name + '.txt'), 'a')
        box = bbox.astype(np.int32)
        # print(box.shape)
        f.write('{} {} {} {} {} {} {} {} {} {} \n'.format(img_id[:-4],
                                              float(score),
                                              int(bbox_obb[0]),
                                              int(bbox_obb[1]),
                                              int(bbox_obb[2]),
                                              int(bbox_obb[3]),
                                              int(bbox_obb[4]),
                                              int(bbox_obb[5]),
                                              int(bbox_obb[6]),
                                              int(bbox_obb[7]),
                                              ))


def show_obb_bbox_result(img, result, class_names, score_thr=0.3, out_file=None):

    assert isinstance(class_names, (tuple, list))
    # print(score_thr)
    img = mmcv.imread(img, flag='unchanged')
    
    if isinstance(result, tuple):
        # bbox_result, segm_result, feature = result
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

       
    #bbox_result 返回的是每个class类别的bbox
    if bbox_result is not None:
        # print('bbox', bbox_result[0].shape)
        # print(bbox_result[0])
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

        labels = np.concatenate(labels)


    bboxes_obb = []
    final_inds = []
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        # print('seg',segms)
        # print(bboxes[:,-1])
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            
            color_mask = np.random.randint(
                0, 256, (3), dtype=np.uint8).tolist()
            mask = maskUtils.decode(segms[i]).astype(np.uint8)
            import cv2
            # print(mask)
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                continue
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            bboxes_obb.append([box[0][0],box[0][1],
                                box[1][0],box[1][1],
                                box[2][0],box[2][1],
                                box[3][0],box[3][1]])
            final_inds.append(i)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[final_inds]
    labels = labels[final_inds]
    bboxes_obb = np.array(bboxes_obb) # 这儿只是坐标,得分取 bboxes 中的得分
    for bbox, bbox_obb, label in zip(bboxes, bboxes_obb, labels):
        class_name = class_names[label]
        score = bbox[-1]
        box = bbox.astype(np.int32)

        pt = []
        pt.append(tuple([bbox_obb[0], bbox_obb[1]]))
        pt.append(tuple([bbox_obb[2], bbox_obb[3]]))
        pt.append(tuple([bbox_obb[4], bbox_obb[5]]))
        pt.append(tuple([bbox_obb[6], bbox_obb[7]]))

        for i in range(4):
                cv2.line(img, pt[i], pt[(i + 1) % 4], color=(0,0,255), thickness=2)
       
        cv2.putText(img, str(class_name)+str(score), (bbox_obb[0], bbox_obb[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('',img)
    cv2.waitKey()

# TODO: merge this method with the one in BaseDetector
def show_rbbox_result(img, result, class_names, score_thr=0.00, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))



    img = mmcv.imread(img, flag='unchanged')
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # # draw segmentation masks
    # if segm_result is not None:
    #     segms = mmcv.concat_list(segm_result)
    #     inds = np.where(bboxes[:, -1] > score_thr)[0]
    #     for i in inds:
    #         color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #         mask = maskUtils.decode(segms[i]).astype(np.bool)
    #         img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 6


    if score_thr > 0:
        assert bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]



    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        x_c, y_c, w, h, theta = bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3], bbox_int[4]
        color = (0, 255, 0)
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, color, 2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))


    plt.imshow(img)
    plt.show()


def show_result(img, result, class_names, score_thr=0.00, out_file=None, mask_postprocess=None):


    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    # print(score_thr)
    img = mmcv.imread(img, flag='unchanged')
    
    if isinstance(result, tuple):
        # bbox_result, segm_result, feature = result
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)



    # draw segmentation masks

    keep_masks = None
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > score_thr
    scores = scores[inds]
    bboxes = bboxes[inds, :]
    labels = labels[inds]


    # print(bboxes.shape)
    for bbox, label, score  in zip(bboxes, labels,scores):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        #cv2.rectangle(
           # img, left_top, right_bottom, (0,255,0), thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(score)
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
    
    # print(feature.size())
    # feature = feature.squeeze()
    # feature = F.softmax(feature)

    # feature = feature[1:,:,:].data.cpu().numpy()

    # feature = np.max(feature, axis=0)
    # feature = mmcv.imresize(feature,(800, 800))
    # plt.subplot(121)
    # plt.imshow(feature)

    # plt.subplot(122)
    # plt.imshow(img)
    # cv2.imshow('img', img)
    # cv2.imwrite('1.png', img)
    # cv2.waitKey()
    # plt.show()
    return img


    
def write_result(output_dir,image_name, img, result, class_names, score_thr=0.00, out_file=None, mask_postprocess=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img, flag='unchanged')
    
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    det_scores = bboxes[:,-1]

    # draw segmentation masks

    keep_masks = None
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(det_scores > score_thr)[0]
        if mask_postprocess is not None:
            masks_result = []
            bboxes_result = []
            for i in inds:
                mask_valid = maskUtils.decode(segms[i])
                bbox_valid = bboxes[i]
                masks_result.append(mask_valid)
                bboxes_result.append(bbox_valid)
            masks_result = np.array(masks_result)
            bboxes_result = np.array(bboxes_result)

            keep_masks = mask_nms(bboxes_result, masks_result)
            for i in keep_masks:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[inds[i]]).astype(np.bool)
                img[mask] = img[mask] * 0.4 + color_mask * 0.6
        else:
            for i in inds:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.4 + color_mask * 0.6


    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5


    if score_thr > 0 and keep_masks is None:
        assert bboxes.shape[1] == 5
        scores = det_scores
        inds = scores > score_thr
        scores = scores[inds]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    else:
        assert bboxes.shape[1] == 5
        scores = det_scores
        inds = scores > score_thr
        scores = scores[inds]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        bboxes = bboxes[keep_masks, :]
        labels = labels[keep_masks]

    for bbox, label,score  in zip(bboxes, labels,scores):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, (0,0,255), thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(score)
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
    print(os.path.join(output_dir,'img_results',image_name))
    # cv2.imwrite(os.path.join(output_dir,'img_results',image_name), img)
