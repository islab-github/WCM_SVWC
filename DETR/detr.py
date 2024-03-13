# %%
import torch
from transformers import AutoImageProcessor, DetaForObjectDetection
from PIL import Image
import requests
import glob

import os
import numpy as np

if os.path.exists(f'./results/') == 0:
    os.mkdir(f'./results/')
if os.path.exists(f'./results/view0/') == 0:
    os.mkdir(f'./results/view0/')
if os.path.exists(f'./results/view1/') == 0:
    os.mkdir(f'./results/view1/')
# %%
# 1. Run this for human detection with view = 0,1
for view in [0, 1]:
    files = glob.glob(f'../data/gs_data_final/view{view}/*/**[!_depth].png',recursive=True)
    if os.path.exists(f'./results/view{view}/output') == 0:
        os.mkdir(f'./results/view{view}/output')
    image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")#.cuda()
    model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large")

    model.eval()
    model = model.cuda()

    with torch.no_grad():
        for i, file_path in enumerate(files):
            image = Image.open(file_path)
            x, y = image.size
            inputs = image_processor(images=image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].cuda()
            inputs['pixel_mask'] = inputs['pixel_mask'].cuda()
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[
                0
            ]
            predicts = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"file:" + file_path.split('.')[-2].split('\\')[-1] + "|"
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                predicts.append((label.item(), *box, score.item()))
            
            person_predicts = []
            for i in range(len(predicts)):
                if predicts[i][0] == 1:
                    per_pre = np.zeros(5)
                    per_pre[0] = predicts[i][1]
                    per_pre[1] = predicts[i][2]
                    per_pre[2] = predicts[i][3]
                    per_pre[3] = predicts[i][4]
                    per_pre[4] = predicts[i][5]
                    person_predicts.append(np.array(per_pre))
            np.savetxt(f'./results/view{view}/output/'+file_path.split(".")[-2].split("\\")[-1]+'.txt',np.array(person_predicts))

        print('-'*30)

# %%
# 2. Run this for cropping with human detection results

import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

for view in [0, 1]:
    directory = f'../data/gs_data_final/view{view}/'
    results_directory = f'./results/view{view}/'
    if os.path.exists(f'{results_directory}cropped_location') == 0:
        os.mkdir(f'{results_directory}cropped_location')
    if os.path.exists(f'{results_directory}cropped_yolo_annotations') == 0:
        os.mkdir(f'{results_directory}cropped_yolo_annotations')
    if os.path.exists(f'{results_directory}cropped_images') == 0:
        os.mkdir(f'{results_directory}cropped_images')
    bbox_data = []
    for i in os.listdir(f'{results_directory}output'):
        data = np.loadtxt(f'{results_directory}output/{i}')
        if len(data.shape) == 1:
            data = np.array([data])
        bbox_data.append(data)
        
    # yolo_annotations_directory = directory + 'yolo_annotations/'
    img_list = glob.glob(directory + '*/**[!_depth].png', recursive=True)
    anno_list = glob.glob(directory + '*/**[!classes].txt', recursive=True)

    pad = 5

    no_object_imgs = []

    for i, img_name in enumerate(img_list):
        img = cv2.imread(img_name)
        y, x, _ = img.shape
        bbox_data_i = bbox_data[i]
        print(bbox_data_i)
        row_list = []
        if len(bbox_data_i) > 0:
            for j, bbox_data_i_j in enumerate(bbox_data_i):
                if len(bbox_data_i_j) == 0:
                    no_object_imgs.append(img_name)
                    continue
                x_left, y_left, x_right, y_right, _ = bbox_data_i_j
                x_left, y_left, x_right, y_right = max(0,int(x_left)-pad), max(0,int(y_left)-pad), min(x,int(x_right)+pad), min(y,int(y_right)+pad)
                new_img = img[y_left:y_right,x_left:x_right,:]
                ratio = min(x/(x_right - x_left),y/(y_right - y_left))
                new_img = cv2.resize(new_img,(0,0),fx=ratio,fy=ratio)
                print(new_img.shape)
                # cv2.imshow('new_img',new_img)
                # cv2.waitKey(0)
                new_y, new_x, _ = new_img.shape
                anno = np.loadtxt(anno_list[i])
                if len(anno.shape) > 1:
                    for row in anno:
                        if row[0] >= 5 and row[1]*x >= x_left and row[1]*x <= x_right and row[2]*y >= y_left and row[2]*y <= y_right:
                            phone_x_cent, phone_y_cent, phone_w, phone_h = row[1]*x, row[2]*y, row[3]*x, row[4]*y
                            ch_phone_x_cent, ch_phone_y_cent, ch_phone_w, ch_phone_h = \
                                (phone_x_cent - x_left) / (x_right - x_left), (phone_y_cent - y_left) / (y_right - y_left), phone_w / (x_right - x_left), phone_h / (y_right - y_left)
                            if ch_phone_x_cent - ch_phone_w/2 < 0:
                                ch_phone_w = ch_phone_w + (ch_phone_x_cent - ch_phone_w/2)
                                ch_phone_x_cent = ch_phone_w/2
                            if ch_phone_y_cent - ch_phone_h/2 < 0:
                                ch_phone_h = ch_phone_h + (ch_phone_y_cent - ch_phone_h/2)
                                ch_phone_y_cent = ch_phone_h/2
                            
                            if row[0] not in row_list:
                                row_list.append(row[0])
                            phone_index = row_list.index(row[0])
                            cv2.imwrite(results_directory+'cropped_images/'+img_name.split('\\')[-1][:-4]+f'_{phone_index}'+'.png', new_img)
                            with open(results_directory+'cropped_yolo_annotations/'+img_name.split('\\')[-1][:-4]+f'_{phone_index}'+'.txt', 'w') as f:
                                line = f"{int(row[0])}" + " " + f'{ch_phone_x_cent} {ch_phone_y_cent} {ch_phone_w} {ch_phone_h}'
                                f.write(line)
                                f.close()
                            
                            with open(results_directory+'cropped_location/'+img_name.split('\\')[-1][:-4]+f'_{phone_index}'+'.txt', 'w') as f:
                                rel_coord = f'{x_left} {y_left} {ratio}'
                                f.write(rel_coord)
                                f.close()

                                    
                                    
                
    print(no_object_imgs)

# %%
# 3. Run this for phone detection with view = 0,1
for view in [0, 1]:
    files = os.listdir(f'./results/view{view}/cropped_images')
    if os.path.exists(f'./results/view{view}/cropped_output') == 0:
        os.mkdir(f'./results/view{view}/cropped_output')
    image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")#.cuda()
    model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large")
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        for i, file_path in enumerate(files):
            image = Image.open(f'./results/view{view}/cropped_images/{file_path}')
            x, y = image.size
            inputs = image_processor(images=image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].cuda()
            inputs['pixel_mask'] = inputs['pixel_mask'].cuda()
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[
                0
            ]
            predicts = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"file: {file_path}|"
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                predicts.append((label.item(), *box, score.item()))
            
            phone_predicts = []
            for i in range(len(predicts)):
                if predicts[i][0] == 77:
                    pho_pre = np.zeros(5)
                    pho_pre[0] = predicts[i][1]
                    pho_pre[1] = predicts[i][2]
                    pho_pre[2] = predicts[i][3]
                    pho_pre[3] = predicts[i][4]
                    pho_pre[4] = predicts[i][5]
                    phone_predicts.append(np.array(pho_pre))
            np.savetxt(f'./results/view{view}/cropped_output/'+file_path.split(".")[-2].split("\\")[-1]+'.txt',np.array(phone_predicts))
        print('-'*30)


#%%
# 4. Run this for formatting the results
from dataclasses import dataclass

import pickle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import glob

from typing import Tuple, List, Dict

directory = f'../data/gs_data_final'
directory_0 = f'../data/gs_data_final/view0'
directory_1 = f'../data/gs_data_final/view1'
results_directory_0 = f'./results/view0'
results_directory_1 = f'./results/view1'

results = {}

iou_thres = 0.3
confi_thres = 0.4

cropped = ''
obj = [0, 4]

bbox_det_data_0 = []
for i in os.listdir(f'{results_directory_0}/output'):
    data = np.loadtxt(f'{results_directory_0}/output/{i}')
    if len(data.shape) == 1:
        data = np.array([data])
    bbox_det_data_0.append(data)

bbox_det_data_1 = []
for i in os.listdir(f'{results_directory_1}/output'):
    data = np.loadtxt(f'{results_directory_1}/output/{i}')
    if len(data.shape) == 1:
        data = np.array([data])
    bbox_det_data_1.append(data)

bbox_det_data = [bbox_det_data_0, bbox_det_data_1]

count_person = np.zeros(2)
for view_num in range(2):
    for i, bbox_image in enumerate(bbox_det_data[view_num]):
        for bbox_det_row in bbox_image:
            if len(bbox_det_row) > 0 and bbox_det_row[-1] > confi_thres:
                count_person[view_num] = count_person[view_num] + 1
    
image_filelist_0 = glob.glob(f'{directory_0}/*/**[!_depth].png', recursive=True)
annotation_filelist_0 = glob.glob(f'{directory_0}/*/**[!classes].txt', recursive=True)
bbox_tru_data_0 = []
for i in range(len(annotation_filelist_0)):
    annotation = np.loadtxt(annotation_filelist_0[i])
    if len(annotation.shape) == 1:
        annotation = np.array([annotation])
    image = cv2.imread(image_filelist_0[i])
    y, x, _ = image.shape
    bbox_tru_data_elem = []
    for row in annotation:
        if row[0] >= obj[0] and row[0] <= obj[1]:
            x_cent_ratio = row[1]
            y_cent_ratio = row[2]
            w_ratio = row[3]
            h_ratio = row[4]
            
            x_left = (x_cent_ratio-w_ratio/2)*x
            y_left = (y_cent_ratio-h_ratio/2)*y
            x_right = (x_cent_ratio+w_ratio/2)*x
            y_right = (y_cent_ratio+h_ratio/2)*y
            
            bbox_tru_data_elem.append(np.array([int(row[0]),x_left,y_left,x_right,y_right]))
                        
    bbox_tru_data_0.append(np.array(bbox_tru_data_elem))

image_filelist_1 = glob.glob(f'{directory_1}/*/**[!_depth].png', recursive=True)
annotation_filelist_1 = glob.glob(f'{directory_1}/*/**[!classes].txt', recursive=True)
bbox_tru_data_1 = []
for i in range(len(annotation_filelist_1)):
    annotation = np.loadtxt(annotation_filelist_1[i])
    if len(annotation.shape) == 1:
        annotation = np.array([annotation])
    image = cv2.imread(image_filelist_1[i])
    y, x, _ = image.shape
    bbox_tru_data_elem = []
    for row in annotation:
        if row[0] >= obj[0] and row[0] <= obj[1]:
            x_cent_ratio = row[1]
            y_cent_ratio = row[2]
            w_ratio = row[3]
            h_ratio = row[4]
            
            x_left = (x_cent_ratio-w_ratio/2)*x
            y_left = (y_cent_ratio-h_ratio/2)*y
            x_right = (x_cent_ratio+w_ratio/2)*x
            y_right = (y_cent_ratio+h_ratio/2)*y
            
            bbox_tru_data_elem.append(np.array([int(row[0]),x_left,y_left,x_right,y_right]))
                        
    bbox_tru_data_1.append(np.array(bbox_tru_data_elem))

bbox_tru_data = [bbox_tru_data_0, bbox_tru_data_1]


for view_num in range(2):

    for i in range(len(bbox_tru_data[view_num])):     # i: image number
        filename = os.listdir(f'./results/view{view_num}/output')[i]
        file_str = filename.split('.txt')[0]
        if file_str not in results:
            results[file_str] = {}

        image_filelist = glob.glob(f'{directory}/view{view_num}/*/**[!_depth].png', recursive=True)
        image = cv2.imread(image_filelist[i])
        y, x, _ = image.shape
        if len(bbox_det_data[view_num][i].shape) == 1:
            bbox_det_data[view_num][i] = np.array([bbox_det_data[view_num][i]])

        # each annotation for person
        ignore_row_list = []
        for bbox_tru_row in bbox_tru_data[view_num][i]:
            person = {}
            TP = False
            num = bbox_tru_row[0]
            bbox_tru = bbox_tru_row[1:]
            area_tru = (bbox_tru[2] - bbox_tru[0])*(bbox_tru[3] - bbox_tru[1])
            tru_abs_pixel = [(bbox_tru[0]+bbox_tru[2])/2,(bbox_tru[1]+bbox_tru[3])/2]
                    
            max_iou = -np.inf
            
            for row_num, bbox_det_row in enumerate(bbox_det_data[view_num][i]):
                if len(bbox_det_row) != 0 and bbox_det_row[-1] >= confi_thres and row_num not in ignore_row_list:
                    bbox_det = bbox_det_row[:4]
                    area_det = (bbox_det[2] - bbox_det[0])*(bbox_det[3] - bbox_det[1])
                    diff_x = min(bbox_det[2],bbox_tru[2])-max(bbox_det[0],bbox_tru[0])
                    diff_y = min(bbox_det[3],bbox_tru[3])-max(bbox_det[1],bbox_tru[1])
                    det_abs_pixel = [(bbox_det[0]+bbox_det[2])/2,(bbox_det[1]+bbox_det[3])/2]
                                
                    if diff_x > 0 and diff_y > 0:
                        area_int = np.max([min(bbox_det[2],bbox_tru[2])-max(bbox_det[0],bbox_tru[0]),0])*np.max([min(bbox_det[3],bbox_tru[3])-max(bbox_det[1],bbox_tru[1]),0])
                        iou = area_int / (area_det + area_tru - area_int)
                        if iou >= iou_thres and iou > max_iou:
                            # print('true positive')
                            TP = True
                            instance = {
                                'y_x_center': [det_abs_pixel[1]/y, det_abs_pixel[0]/x],
                                'gt_y_x_center': [tru_abs_pixel[1]/y, tru_abs_pixel[0]/x],
                                'width': (bbox_det[2]-bbox_det[0])/x, 'length': (bbox_det[3]-bbox_det[1])/y,
                                'gt_width': (bbox_tru[2]-bbox_tru[0])/x, 'gt_length': (bbox_tru[3]-bbox_tru[1])/y,
                                'is_detected': 1.0, 'is_available': 1.0
                            }
                            max_row_num = row_num
                            max_iou = iou

            if TP == True:
                ignore_row_list.append(max_row_num)
            
            
                        
            if TP == False:
                instance = {
                    'y_x_center': [0.0, 0.0],
                    'gt_y_x_center': [tru_abs_pixel[1]/y, tru_abs_pixel[0]/x],
                    'width': 0.0, 'length': 0.0,
                    'gt_width': (bbox_tru[2]-bbox_tru[0])/x, 'gt_length': (bbox_tru[3]-bbox_tru[1])/y,
                    'is_detected': 0.0, 'is_available': 1.0
                }
            # a[f'person{int(num)+1}'][f'view{view_num}'] = instance
            if f'person{int(num)+1}' not in results[file_str]:
                results[file_str][f'person{int(num)+1}'] = {'view0': {
                    'y_x_center': [0.0, 0.0],
                    'gt_y_x_center': [0.0, 0.0],
                    'width': 0.0, 'length': 0.0,
                    'gt_width': 0.0, 'gt_length': 0.0,
                    'is_detected': 0.0, 'is_available': 0.0
                }, 'view1': {
                    'y_x_center': [0.0, 0.0],
                    'gt_y_x_center': [0.0, 0.0],
                    'width': 0.0, 'length': 0.0,
                    'gt_width': 0.0, 'gt_length': 0.0,
                    'is_detected': 0.0, 'is_available': 0.0
                }}
                
            if f'view{view_num}' not in results[file_str][f'person{int(num)+1}']:
                results[file_str][f'person{int(num)+1}'][f'view{view_num}'] = {}
            results[file_str][f'person{int(num)+1}'][f'view{view_num}'] = instance

cropped = 'cropped_'
obj = [5, 9]
iou_thres_phone = -np.inf
iou_thres = 0.01
confi_thres = 0.01
bbox_det_data_0 = []
for i in os.listdir(f'{results_directory_0}/{cropped}output'):
    data = np.loadtxt(f'{results_directory_0}/{cropped}output/{i}')
    if len(data.shape) > 1 and data.shape[0] > 3:
        data = data[np.argpartition(data[:, -1], kth = -1)[-1:]]
    # print(i)
    if len(data.shape) == 1:
        data = np.array([data])
    bbox_det_data_0.append(data)

bbox_det_data_1 = []
for i in os.listdir(f'{results_directory_1}/{cropped}output'):
    data = np.loadtxt(f'{results_directory_1}/{cropped}output/{i}')
    if len(data.shape) > 1 and data.shape[0] > 3:
        data = data[np.argpartition(data[:, -1], kth = -1)[-1:]]
    if len(data.shape) == 1:
        data = np.array([data])
    bbox_det_data_1.append(data)

bbox_det_data = [bbox_det_data_0, bbox_det_data_1]

count_phone = np.zeros(2)
for view_num in range(2):
    for i, bbox_image in enumerate(bbox_det_data[view_num]):
        for bbox_det_row in bbox_image:
            if len(bbox_det_row) > 0 and bbox_det_row[-1] > confi_thres:
                count_phone[view_num] = count_phone[view_num] + 1
                break

    


    
image_filelist_0 = os.listdir(f'{results_directory_0}/{cropped}images/')
annotation_filelist_0 = os.listdir(f'{results_directory_0}/{cropped}yolo_annotations/')
bbox_tru_data_0 = []
for i in range(len(annotation_filelist_0)):
    annotation = np.loadtxt(f'{results_directory_0}/{cropped}yolo_annotations/' + annotation_filelist_0[i])
    if len(annotation.shape) == 1:
        annotation = np.array([annotation])
    image = cv2.imread(f'{results_directory_0}/{cropped}images/' + image_filelist_0[i])
    # image_show = Image.open((directory+'cropped_images/'+image_filelist[i][:-4]+f'_{j}'+'.png'))
    y, x, _ = image.shape
    bbox_tru_data_elem = []
    for row in annotation:
        if row[0] >= obj[0] and row[0] <= obj[1]:
            x_cent_ratio = row[1]
            y_cent_ratio = row[2]
            w_ratio = row[3]
            h_ratio = row[4]
            
            x_left = (x_cent_ratio-w_ratio/2)*x
            y_left = (y_cent_ratio-h_ratio/2)*y
            x_right = (x_cent_ratio+w_ratio/2)*x
            y_right = (y_cent_ratio+h_ratio/2)*y
            
            bbox_tru_data_elem.append(np.array([int(row[0]),x_left,y_left,x_right,y_right]))
                        
    bbox_tru_data_0.append(np.array(bbox_tru_data_elem))

image_filelist_1 = os.listdir(f'{results_directory_1}/{cropped}images/')
annotation_filelist_1 = os.listdir(f'{results_directory_1}/{cropped}yolo_annotations/')
bbox_tru_data_1 = []
for i in range(len(annotation_filelist_1)):
    annotation = np.loadtxt(f'{results_directory_1}/{cropped}yolo_annotations/' + annotation_filelist_1[i])
    if len(annotation.shape) == 1:
        annotation = np.array([annotation])
    image = cv2.imread(f'{results_directory_1}/{cropped}images/' + image_filelist_1[i])
    # image_show = Image.open((directory+'cropped_images/'+image_filelist[i][:-4]+f'_{j}'+'.png'))
    y, x, _ = image.shape
    bbox_tru_data_elem = []
    for row in annotation:
        if row[0] >= obj[0] and row[0] <= obj[1]:
            x_cent_ratio = row[1]
            y_cent_ratio = row[2]
            w_ratio = row[3]
            h_ratio = row[4]
            
            x_left = (x_cent_ratio-w_ratio/2)*x
            y_left = (y_cent_ratio-h_ratio/2)*y
            x_right = (x_cent_ratio+w_ratio/2)*x
            y_right = (y_cent_ratio+h_ratio/2)*y
            
            bbox_tru_data_elem.append(np.array([int(row[0]),x_left,y_left,x_right,y_right]))
                        
    bbox_tru_data_1.append(np.array(bbox_tru_data_elem))

bbox_tru_data = [bbox_tru_data_0, bbox_tru_data_1]


# confi_thres = 0.5
for view_num in range(2):
    det = np.ones([len(bbox_det_data[view_num]),5])*(-1)
    tru = np.zeros([len(bbox_tru_data[view_num]),5])
    for i in range(len(bbox_tru_data[view_num])):     # i: image number
        filename = os.listdir(f'./results/view{view_num}/cropped_output')[i]
        file_str = filename[:-1-len(filename.split('_')[-1])]
        if file_str not in results:
            results[file_str] = {}
        image_filelist = os.listdir(f'./results/view{view_num}/cropped_images/')
        image_name = image_filelist[i][:-1-len(image_filelist[i].split('_')[-1])] + '.png'
        abs_image_name = glob.glob(f'../data/gs_data_final/view{view_num}/*/{image_name}', recursive=True)[0]
        image = cv2.imread(abs_image_name)
        
        # image_show = Image.open((directory+'cropped_images/'+image_filelist[i][:-4]+f'_{j}'+'.png'))
        y, x, _ = image.shape
        if len(bbox_det_data[view_num][i].shape) == 1:
            bbox_det_data[view_num][i] = np.array([bbox_det_data[view_num][i]])

        ignore_row_list = []
        for bbox_tru_row in bbox_tru_data[view_num][i]:
            person = {}
            TP = False
            num = bbox_tru_row[0]
            bbox_tru = bbox_tru_row[1:]
            area_tru = (bbox_tru[2] - bbox_tru[0])*(bbox_tru[3] - bbox_tru[1])
            max_iou = -np.inf
            for row_num, bbox_det_row in enumerate(bbox_det_data[view_num][i]):
                if len(bbox_det_row) != 0 and bbox_det_row[-1] >= confi_thres and row_num not in ignore_row_list:
                    
                    bbox_det = bbox_det_row[:4]
                    area_det = (bbox_det[2] - bbox_det[0])*(bbox_det[3] - bbox_det[1])
                    diff_x = min(bbox_det[2],bbox_tru[2])-max(bbox_det[0],bbox_tru[0])
                    diff_y = min(bbox_det[3],bbox_tru[3])-max(bbox_det[1],bbox_tru[1])
                    location = np.loadtxt(f'./results/view{view_num}/cropped_location/'+image_filelist[i][:-4]+'.txt')
                    det_abs_pixel = [(bbox_det[0]+bbox_det[2])/2/location[2]+location[0],(bbox_det[1]+bbox_det[3])/2/location[2]+location[1]]
                    tru_abs_pixel = [(bbox_tru[0]+bbox_tru[2])/2/location[2]+location[0],(bbox_tru[1]+bbox_tru[3])/2/location[2]+location[1]]              
                                
                    if diff_x > 0 and diff_y > 0:
                        area_int = np.max([min(bbox_det[2],bbox_tru[2])-max(bbox_det[0],bbox_tru[0]),0])*np.max([min(bbox_det[3],bbox_tru[3])-max(bbox_det[1],bbox_tru[1]),0])
                        iou = area_int / (area_det + area_tru - area_int)
                        if iou >= iou_thres_phone and iou > max_iou:
                            # print('true positive')
                            TP = True
                            instance = {'y_x_center': [det_abs_pixel[1]/y, det_abs_pixel[0]/x], 'gt_y_x_center': [tru_abs_pixel[1]/y, tru_abs_pixel[0]/x],
                                                            'width': (bbox_det[2]-bbox_det[0])/x, 'length': (bbox_det[3]-bbox_det[1])/y,
                                                            'gt_width': (bbox_tru[2]-bbox_tru[0])/x, 'gt_length': (bbox_tru[3]-bbox_tru[1])/y,
                                                            'is_detected': 1.0, 'is_available': 1.0}
                            max_row_num = row_num
                            max_iou = iou
            
            if TP == True:
                ignore_row_list.append(max_row_num)

            if TP == False:
                location = np.loadtxt(f'./results/view{view_num}/cropped_location/'+image_filelist[i][:-4]+'.txt')
                tru_abs_pixel = [(bbox_tru[0]+bbox_tru[2])/2/location[2]+location[0],(bbox_tru[1]+bbox_tru[3])/2/location[2]+location[1]]
                
                instance = {'y_x_center': [0.0, 0.0], 'gt_y_x_center': [tru_abs_pixel[1]/y, tru_abs_pixel[0]/x],
                                                        'width': 0.0, 'length': 0.0,
                                                        'gt_width': (bbox_tru[2]-bbox_tru[0])/x, 'gt_length': (bbox_tru[3]-bbox_tru[1])/y,
                                                        'is_detected': 0.0, 'is_available': 1.0}
            if f'phone{int(num)+1-5}' not in results[file_str]:
                results[file_str][f'phone{int(num)+1-5}'] = {'view0': {
                    'y_x_center': [0.0, 0.0],
                    'gt_y_x_center': [0.0, 0.0],
                    'width': 0.0, 'length': 0.0,
                    'gt_width': 0.0, 'gt_length': 0.0,
                    'is_detected': 0.0, 'is_available': 0.0
                }, 'view1': {
                    'y_x_center': [0.0, 0.0],
                    'gt_y_x_center': [0.0, 0.0],
                    'width': 0.0, 'length': 0.0,
                    'gt_width': 0.0, 'gt_length': 0.0,
                    'is_detected': 0.0, 'is_available': 0.0
                }}
                
            if f'view{view_num}' not in results[file_str][f'phone{int(num)+1-5}']:
                results[file_str][f'phone{int(num)+1-5}'][f'view{view_num}'] = {}
            results[file_str][f'phone{int(num)+1-5}'][f'view{view_num}'] = instance
                

results['num_person'] = list(count_person)
results['num_phone'] = list(count_phone)


with open(f'./results.json', 'w') as f:
    json.dump(results, f, indent = 2)
