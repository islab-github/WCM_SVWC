import numpy as np
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
    if (len(bbox_gt) == 0) or (len(bbox_pred) == 0):
        return [], [], []

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


def non_max_suppression_fast(boxes, scores=None, iou_threshold: float = 0.1):
    """
    boxes : coordinates of each box
    scores : score of each box
    iou_threshold : iou threshold(box with iou larger than threshold will be removed)
    """
    if len(boxes) == 0:
        return [], []
    if len(boxes) == 1:
        return boxes, [0]

    # Init the picked box info
    pick = []

    # Box coordinate consist of left top and right bottom
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Greedily select the order of box to compare iou
    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)
    indices = np.argsort(scores)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[0]
        pick.append(i)

        # With vector implementation, we can calculate fast
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h

        # Calculate the iou
        iou = intersection / (area[indices[:last]] + area[indices[last]] - intersection)
        
        # if IoU of a box is larger than threshold, remove that box index.
        indices = np.delete(indices, np.concatenate(([0], np.where(iou > iou_threshold)[0])))

    pick = list(set(pick))
    return boxes[pick], pick


def nms_class_wise(boxes, iou_threshold, scores=None, obj_cls=None, find_class=None):
    """
    boxes : coordinates of each box
    scores : score of each box
    iou_threshold : iou threshold(box with iou larger than threshold will be removed)
    """
    if not(len(boxes) > 0):
        return [], [], []

    s_idx = obj_cls == obj_list.index(find_class)
    boxes, pick = non_max_suppression_fast(
        boxes[s_idx], scores[s_idx], iou_threshold
    )
    return boxes, scores[s_idx][pick], obj_cls[s_idx][pick]

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

from .data_utils import standard_to_bgr, STANDARD_COLORS

color_list = standard_to_bgr(STANDARD_COLORS)
