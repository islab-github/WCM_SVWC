import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


def bbox_iou(box1, box2) -> float:
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_x = x2_min - x1_max + 1
    inter_y = y2_min - y1_max + 1

    if (inter_x <= 0) or (inter_y <= 0):
        return -1.0

    inter = inter_x * inter_y
    union = box1_area + box2_area - inter
    iou = float(inter / union)
    return iou


def match_bboxes(bbox_gt, bbox_pred, threshold=0.3):
    """
    https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.
    threshold: IOU threshold

    Returns
    -------
    (idxs_true, idxs_pred, ious)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
    """
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i], bbox_pred[j])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix, np.zeros((diff, n_pred))), axis=0)
    elif n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix, np.zeros((n_true, diff))), axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = linear_sum_assignment(1 - iou_matrix)

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > threshold)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid]
