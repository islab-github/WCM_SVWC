#%%
# -*- coding: utf-8 -*-
"""
inference + evaluate together
NOTE THAT LABEL FILE IS IN TXT, YOLO LABEL
"""
import os.path
from collections import OrderedDict
import time
import json
from tqdm import tqdm

from convert_pred import convert_pred
from sum_json import sum_json
from load_test_wirelessmag import load_data_view_wj

import numpy as np
import torch
import cv2
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.data_utils import preprocess, postprocess, plot_one_box, get_index_label, invert_affine
from utils.dectection_util import obj_list, color_list, nms_class_wise

MODEL_TYPE = 8
TARGET = "phone"

ROOT_DIR ="../data/gs_data_final"
MODEL_CKPT = f"../data/efficientdet-d{MODEL_TYPE}.pth"

PERSON_ID = 0
PHONE_ID = 76
PAD_RATIO = 0.3


def count_objects(file_path, threshold_person=90, threshold_cell_phone=40):

    with open(file_path, 'r') as file:
        data = json.load(file)

    person_count = 0
    cell_phone_count = 0

    for key in data.keys():
        annotations = data[key]
        for annotation in annotations.values():
            obj = annotation.get("obj", "")
            score = annotation.get("score", 0)

            if obj == "person" and score >= threshold_person:
                person_count += 1
            elif obj == "cell phone" and score >= threshold_cell_phone:
                cell_phone_count += 1

    return person_count, cell_phone_count

def load_model(model_type: int = MODEL_TYPE):
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    model = EfficientDetBackbone(
        compound_coef=MODEL_TYPE, num_classes=90,
        anchor_ratios=anchor_ratios, anchor_scales=anchor_scales
    )
    model.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu"), strict=True)
    model = model.cuda()
    model.eval()
    return model, input_sizes[model_type]

def detect(framed_metas, model, img, threshold: float, iou_threshold: float):
    start_time = time.time()         
    features, regression, classification, anchors = model(img)
    total_time = time.time() - start_time

    RegressB = BBoxTransform()
    ClipB = ClipBoxes()

    out = postprocess(
        img,
        anchors, regression, classification,
        RegressB, ClipB,
        threshold, iou_threshold
    )
    out = invert_affine(framed_metas, out)
    return out, total_time


def display(pred, img, img_key: str, display_objects=("person", "cell phone"), save: bool = False):
    img = img.copy()
    for j in range(len(pred['rois'])):
        x1, y1, x2, y2 = pred['rois'][j].astype(np.int32)
        obj = obj_list[pred['class_ids'][j]]
        if obj in display_objects:
            score = float(pred['scores'][j])
            if score > 0.5:
               plot_one_box(img, [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

    if save:
        cv2.imwrite(f"testMultiview0/{img_key}", img)


def crop(img, box, pad_ratio: float = PAD_RATIO):
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = (x2 - x1), (y2 - y1)
    pad_w = (w * pad_ratio) // 2
    pad_h = (h * pad_ratio) // 2
    pad = max(pad_w, pad_h)
    new_x1, new_y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    new_x2, new_y2 = int(min(x2 + pad, img_w)), int(min(y2 + pad, img_h))

    crop_img = img[new_y1:new_y2, new_x1:new_x2, :].copy()
    return crop_img, (new_x1, new_y1, new_x2, new_y2)

def boundingbox(x, y, l, h):
    x1 = round((x-l/2) * 640)
    y1 = round((y-h/2) * 480)
    x2 = round((x+l/2) * 640)
    y2 = round((y+h/2) * 480)
    return x1, y1, x2, y2


def run(data):
    model, input_size = load_model(model_type=MODEL_TYPE)
    torch.set_grad_enabled(False)

    json_prediction = OrderedDict()

    TOTAL_TIME = 0
    total_samples = 0
    for count, (rgb_key, (gt_path, rgb_path, _,_,_)) in tqdm(enumerate(data.items()), total= len(data.items())):
        # print(f"{count} / {len(data)}, {rgb_key}")
        ori_img, framed_img, framed_metas = preprocess(rgb_path, max_size=input_size)

        img_th = torch.from_numpy(framed_img[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)
        
        prediction, t1 = detect(
            framed_metas, model, img_th,
            threshold=0.3,
            iou_threshold=0.5
        )
        prediction = prediction[0]

        count = 0
        TOTAL_TIME_SAMPLE = 0
        prediction2 = {'rois': np.array([]), 'scores': np.array([]), 'class_ids': np.array([])}
        for j in range(len(prediction['rois'])):
            obj = obj_list[prediction['class_ids'][j]]
            if obj == "person":
                count += 1
                x1, y1, x2, y2 = prediction['rois'][j].astype(np.int32)
                crop_img, (crop_x1, crop_y1, crop_x2, crop_y2) = crop(ori_img[0], (x1, y1, x2, y2), pad_ratio=PAD_RATIO)
                if crop_img.size == 0:
                    continue

                crop_path = rgb_path.replace(rgb_key, f"crop\\{rgb_key[:-4] + 'crop' + str(j) + '.png'}")

                if not os.path.exists(rgb_path.replace(rgb_key, "crop")):
                    os.makedirs(rgb_path.replace(rgb_key, "crop"))
                cv2.imwrite(crop_path, crop_img)

                #cropped image is the input
                _, framed_img, framed_metas = preprocess(crop_path, max_size=input_size)
                img_th = torch.from_numpy(framed_img[0]).cuda().unsqueeze(0).permute(0, 3, 1, 2)

                prediction2, t2 = detect(
                    framed_metas, model, img_th,
                    threshold=0.01,
                    iou_threshold=0.3
                )
                prediction2 = prediction2[0]
                # print("--- %s seconds ---" % (t2))
                TOTAL_TIME_SAMPLE += t2
                
                if len(prediction2["rois"]) > 0:
                    prediction2["rois"][:, 0] += crop_x1
                    prediction2["rois"][:, 1] += crop_y1
                    prediction2["rois"][:, 2] += crop_x1
                    prediction2["rois"][:, 3] += crop_y1

                    prediction["rois"] = np.concatenate((prediction["rois"], prediction2["rois"]))
                    prediction["class_ids"] = np.concatenate((prediction["class_ids"], prediction2["class_ids"]))
                    prediction["scores"] = np.concatenate((prediction["scores"], prediction2["scores"]))

        if count != 0:
            TOTAL_TIME += TOTAL_TIME_SAMPLE/count     
            total_samples += 1

        display(prediction, ori_img[0], rgb_key, display_objects=("person", "cell phone"), save=True)

        if rgb_key in json_prediction.keys():
            raise ValueError(f"data key error, {rgb_key}")
        json_prediction[rgb_key] = OrderedDict()

        pred_person = nms_class_wise(
            prediction['rois'],
            scores=prediction['scores'],
            obj_cls=prediction['class_ids'],
            find_class="person",
            iou_threshold=0.1
        )

        pred_phone = nms_class_wise(
            prediction['rois'],
            scores=prediction['scores'],
            obj_cls=prediction['class_ids'],
            find_class="cell phone",
            iou_threshold=0.2
        )

        json_count = 0

        for xy, s in zip(pred_person[0], pred_person[1]):
            x1, y1, x2, y2 = map(int, xy)
            json_prediction[rgb_key][str(json_count)] = OrderedDict(
                obj="person", x1=x1, y1=y1, x2=x2, y2=y2,
                score=int(s * 100))
            json_count += 1

        for xy, s in zip(pred_phone[0], pred_phone[1]):
            x1, y1, x2, y2 = map(int, xy)
            json_prediction[rgb_key][str(json_count)] = OrderedDict(
                obj="cell phone", x1=x1, y1=y1, x2=x2, y2=y2,
                score=int(s * 100))
            json_count += 1


    with open(JSON_PRED, "w") as f:
       json.dump(json_prediction, f)


if __name__ == '__main__':

    person_count = [0, 0]
    cell_phone_count = [0, 0]
    threshold_person = 70
    threshold_cell_phone = 10

    for v in [0, 1]:

        DATA_DIR = ROOT_DIR + f"\\view{v}"
        JSON_PRED = ROOT_DIR + f"\\predictionVIEWPOINT{v}.json"

        data = load_data_view_wj(DATA_DIR)
        # if not os.path.isfile(JSON_PRED):
        #     run(data)
        run(data)

        person_count[v], cell_phone_count[v] = count_objects(JSON_PRED, threshold_person, threshold_cell_phone)

        files = os.listdir(DATA_DIR)

        for f in files:
            if f == ".DS_Store":
                continue

            args = {
                "File_path": DATA_DIR + "/" + f,
                "File_name": f,
                "result_file": JSON_PRED,
                "view": v,
                "threshold_person": threshold_person,
                "threshold_phone": threshold_cell_phone
            }
            convert_pred(args)

    files = os.listdir(DATA_DIR)
    output = {
        "num_person": person_count,
        "num_phone": cell_phone_count
    }

    for f in files:
        if f == ".DS_Store":
            continue

        args = {
            "File_path": ROOT_DIR + f"\\view{0}" + "/" + f,
            "File_path2": ROOT_DIR + f"\\view{1}" + "/" + f,
            "File_name": f
        }

        current_output = sum_json(args)
        output.update(current_output)

    with open(f'./multiview_measurement.json', 'w') as f:
        json.dump(output, f, indent=4)
