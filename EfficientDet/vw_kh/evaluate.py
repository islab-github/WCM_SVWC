import numpy as np
from load import load_data, load_prediction
from parse import parse_xml, parse_prediction, parse_distance, mean_distance, center_angle
from utils import match_bboxes

data = load_data()
prediction = load_prediction()
angle = np.load("degree.npy")
assert len(data) == len(prediction)
total = len(data)

tp = 0
fp = 0
fn = 0
dist_err = 0.0
dist_valid_count = 0

angle_err = 0.0
angle_valid_count = 0

# already sorted
for count, ((gt_path, rgb_path, distance_path), pred) in enumerate(zip(data.values(), prediction.values())):
    print(f"{count} / {total}, {rgb_path}")

    dist = parse_distance(distance_path)
    box_gt = np.array(parse_xml(gt_path))
    box_pred = np.array(parse_prediction(pred))

    idx_gt, idx_pred, iou_score = match_bboxes(box_gt, box_pred, threshold=0.1)

    num_gt = len(box_gt)
    num_pred = len(box_pred)
    num_match = len(idx_gt)

    tp += num_match
    fn += (num_gt - num_match)
    fp += (num_pred - num_match)

    for ig, ip in zip(idx_gt, idx_pred):
        bg = box_gt[ig]
        bp = box_pred[ip]

        dist_bg = mean_distance(dist, bg)
        dist_bp = mean_distance(dist, bp)
        if (dist_bg > 0) and (dist_bp > 0):
            dist_err += abs(dist_bg - dist_bp)
            dist_valid_count += 1

        angle_bg = center_angle(angle, bg)
        angle_bp = center_angle(angle, bp)
        if (angle_bg > 0) and (angle_bp > 0):
            angle_err += abs(angle_bg - angle_bp)
            angle_valid_count += 1

    # print(box_gt)
    # print(box_pred)
    # print(idx_gt)
    # print(idx_pred)
    # print(iou_score)

    # if count > 20:
    #     break

precision = tp / (tp + fp)
recall = tp / (tp + fn)
mean_dist_err = dist_err / max(dist_valid_count, 1)
mean_angle_err = angle_err / max(angle_valid_count, 1)

print(f"Precision: {precision:.6f} (TP: {tp}, FP: {fp})")
print(f"Recall: {recall:.6f} (TP: {tp}, FN: {fn})")
print(f"Distance error: {mean_dist_err * 100:6f} cm (valid: {dist_valid_count})")
print(f"Angle error: {mean_angle_err:6f} degree (valid: {angle_valid_count})")
