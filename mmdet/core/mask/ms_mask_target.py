import mmcv
import torch
import numpy as np


def ms_mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    target = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    target = list(target)
    
    s0_mask_targets = torch.cat([target[0][0], target[1][0]])
    s1_mask_targets = torch.cat([target[0][1], target[1][1]])
    s2_mask_targets = torch.cat([target[0][2], target[1][2]])
# 
    # s0_mask_targets = target[0][0]
    # s1_mask_targets = target[0][1]
    # s2_mask_targets = target[0][2]

    return s0_mask_targets, s1_mask_targets, s2_mask_targets



def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    s0_mask_targets = []
    s1_mask_targets = []
    s2_mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            s0_target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size, mask_size))

            s1_target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size//2, mask_size//2))

            s2_target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   (mask_size//4, mask_size//4))
         
            s0_mask_targets.append(s0_target)
            s1_mask_targets.append(s1_target)
            s2_mask_targets.append(s2_target)

        s0_mask_targets = torch.from_numpy(np.stack(s0_mask_targets)).float().to(
            pos_proposals.device)
        s1_mask_targets = torch.from_numpy(np.stack(s1_mask_targets)).float().to(
            pos_proposals.device)
        s2_mask_targets = torch.from_numpy(np.stack(s2_mask_targets)).float().to(
            pos_proposals.device)
    else:
        s0_mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
        s1_mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
        s2_mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    

    return s0_mask_targets, s1_mask_targets, s2_mask_targets
