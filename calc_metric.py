#%%
from pprint import pprint

from coordinates import MultiViewCoordinate
import numpy as np
import json
import glob
import copy
import argparse
from tqdm.auto import tqdm

import pandas as pd
from IPython.display import HTML

def distance_error(gt, pd):
    return np.sqrt(np.sum(np.square(gt - pd), axis=-1))
def angle_error(gt, pd):
    return np.abs(gt - pd)

if __name__ == '__main__':
    # Just get the name of places
    path_view0 = "./data/gs_data_final/view0/"
    path_view1 = "./data/gs_data_final/view1/"

    place = json.load(open("./data/adjusted_measurement_angles_v3.json"))

    degree_file = "./data/degree.npy"
    h, w = 480, 640

    box_file = f"./DETR/results.json"
    gt_file = f"./DETR/results.json"

    box = json.load(open(box_file))
    box_gt = json.load(open(gt_file))

    place_to_save = copy.deepcopy(place)
    min_distance_to_save = {}
    results = []
    results.append({
        "single_az_error_v0"   :[],
        "single_el_error_v0"   :[],
        "single_az_error_v1"   :[],
        "single_el_error_v1"   :[],
        "single_distance_error_v0":[],
        "single_distance_error_v1":[],
        "is_available_v0"   :[],
        "is_available_v1"   :[],
        "is_detected_v0"    :[],
        "is_detected_v1"    :[],
        "multi_el_error" :[],
        "multi_az_error" :[],
        "multi_distance_error"    :[],
    })

    results.append({
        "single_az_error_v0"   :[],
        "single_el_error_v0"   :[],
        "single_az_error_v1"   :[],
        "single_el_error_v1"   :[],
        "single_distance_error_v0":[],
        "single_distance_error_v1":[],
        "is_available_v0"   :[],
        "is_available_v1"   :[],
        "is_detected_v0"    :[],
        "is_detected_v1"    :[],
        "multi_el_error" :[],
        "multi_az_error" :[],
        "multi_distance_error"    :[],
    })
    detected_person = 0
    detected_phone = 0
    for pp in range(2):
        for p in tqdm(place.keys(), desc=f'Calc Metric for : ', leave=False):
            files_view0 = glob.glob(path_view0 + p + '/' + '*.json')
            files_view1 = glob.glob(path_view1 + p + '/' + '*.json')

            files_view0.sort()
            files_view1.sort()

            View = MultiViewCoordinate(**place[p], degree_file0 = degree_file)

            for f_v0 in files_view0:
                if "convert" in f_v0:
                    continue
                file_index = f_v0.split("_")[-1].split(".")[0]

                d_v0 = json.load(open(f_v0))
                depth_v0 = np.array(d_v0)

                f_v1 = path_view1 + p + '/' + p + '_' + file_index + '.json'
                d_v1 = json.load(open(f_v1))
                depth_v1 = np.array(d_v1)

                # Box info (GT and PD)
                b = box[f"{p}_{file_index}"]
                b_gt = box_gt[f"{p}_{file_index}"]

                b_v0_gt = []
                b_v1_gt = []

                b_v0_pd = []
                b_v1_pd = []

                obj_detected_v0 = []
                obj_detected_v1 = []

                obj_available_v0 = []
                obj_available_v1 = []

                # Object info (GT and PD)
                for k in b_gt.keys():
                    if (pp == 0 and k[:-1] == 'person') or (pp == 1 and k[:-1] == 'phone'):
                        if k not in b.keys():
                            continue
                        # ground truth
                        b_v0_gt.append(
                            b_gt[k]["view0"]["gt_y_x_center"]
                        )
                        b_v1_gt.append(
                            b_gt[k]["view1"]["gt_y_x_center"]
                        )

                        # prediction
                        b_v0_pd.append(
                            b[k]["view0"]["y_x_center"]
                        )
                        b_v1_pd.append(
                            b[k]["view1"]["y_x_center"]
                        )

                        obj_detected_v0.append(b[k]["view0"]["is_detected"])
                        obj_detected_v1.append(b[k]["view1"]["is_detected"])

                        obj_available_v0.append(b[k]["view0"]["is_available"])
                        obj_available_v1.append(b[k]["view1"]["is_available"])

                # is box detected & available in the view
                results[pp]["is_detected_v0"] += obj_detected_v0
                results[pp]["is_detected_v1"] += obj_detected_v1

                results[pp]["is_available_v0"] += obj_available_v0
                results[pp]["is_available_v1"] += obj_available_v1

                xyz_gt_0, xyz_gt_1 = View.to_shared_coordinates(
                    np.array(b_v0_gt),
                    np.array(b_v1_gt),
                    depth_v0,
                    depth_v1
                )

                xyz_pd_0, xyz_pd_1 = View.to_shared_coordinates(
                    np.array(b_v0_pd),
                    np.array(b_v1_pd),
                    depth_v0,
                    depth_v1
                )

                b_v0_gt = np.array(b_v0_gt)
                b_v1_gt = np.array(b_v1_gt)

                b_v0_pd = np.array(b_v0_pd)
                b_v1_pd = np.array(b_v1_pd)

                single_angle_gt0 = View.cam0.to_angle(np.rint(b_v0_gt * np.array([[480,640]])).astype(int))
                single_angle_gt1 = View.cam1.to_angle(np.rint(b_v1_gt * np.array([[480,640]])).astype(int))

                single_angle_pd0 = View.cam0.to_angle(np.rint(b_v0_pd * np.array([[480,640]])).astype(int))
                single_angle_pd1 = View.cam1.to_angle(np.rint(b_v1_pd * np.array([[480,640]])).astype(int))

                single_angle_error0 = angle_error(np.array(single_angle_gt0).transpose(),
                                                np.array(single_angle_pd0).transpose())
                single_angle_error1 = angle_error(np.array(single_angle_gt1).transpose(),
                                                np.array(single_angle_pd1).transpose())

                single_distance_error0 = distance_error(xyz_gt_0, xyz_pd_0)
                single_distance_error1 = distance_error(xyz_gt_1, xyz_pd_1)

                m_a_s0 = single_angle_error0.copy()
                m_a_s1 = single_angle_error1.copy()

                m_a_s0[np.array(obj_detected_v0)==0,:] = 1e5
                m_a_s1[np.array(obj_detected_v1)==0,:] = 1e5

                m_d_s0 = single_distance_error0.copy()
                m_d_s1 = single_distance_error1.copy()

                m_d_s0[np.array(obj_detected_v0)==0] = 1e5
                m_d_s1[np.array(obj_detected_v1)==0] = 1e5

                multi_el_error = np.min([
                    m_a_s0[:,0],
                    m_a_s1[:,0]
                ], 0).flatten().tolist()
                multi_az_error = np.min([
                    m_a_s0[:,1],
                    m_a_s1[:,1]
                ], 0).flatten().tolist()
                multi_distance_error = np.min([
                    m_d_s0,
                    m_d_s1
                ], 0).flatten().tolist()

                # Single-view
                results[pp]["single_el_error_v0"] = results[pp]["single_el_error_v0"] + single_angle_error0[:,0].flatten().tolist()
                results[pp]["single_el_error_v1"] = results[pp]["single_el_error_v1"] + single_angle_error1[:,0].flatten().tolist()

                results[pp]["single_az_error_v0"] = results[pp]["single_az_error_v0"] + single_angle_error0[:,1].flatten().tolist()
                results[pp]["single_az_error_v1"] = results[pp]["single_az_error_v1"] + single_angle_error1[:,1].flatten().tolist()

                results[pp]["single_distance_error_v0"] = results[pp]["single_distance_error_v0"] + single_distance_error0.flatten().tolist()
                results[pp]["single_distance_error_v1"] = results[pp]["single_distance_error_v1"] + single_distance_error1.flatten().tolist()

                # Multi-view
                results[pp]["multi_el_error"] = results[pp]["multi_el_error"] + multi_el_error
                results[pp]["multi_az_error"] = results[pp]["multi_az_error"] + multi_az_error
                results[pp]["multi_distance_error"] = results[pp]["multi_distance_error"] + multi_distance_error


    count_person = np.array(box["num_person"])
    count_phone = np.array(box["num_phone"])

    # person results
    single_v0_person_recall = np.sum(results[0]["is_detected_v0"]) / len(results[0]["is_available_v0"])
    single_v1_person_recall = np.sum(results[0]["is_detected_v1"]) / len(results[0]["is_available_v1"])

    multi_single_v0_person_recall = np.sum(results[0]["is_detected_v0"]) / len(results[0]["is_available_v0"])
    multi_single_v1_person_recall = np.sum(results[0]["is_detected_v1"]) / len(results[0]["is_available_v1"])

    multi_person_recall = np.count_nonzero(
        np.array(results[0]["is_detected_v0"]) + np.array(results[0]["is_detected_v1"])
    ) / len(results[0]["is_available_v1"])

    single_v0_person_precision = np.count_nonzero(
        np.array(results[0]["is_detected_v0"])
    ) / count_person[0]
    single_v1_person_precision = np.count_nonzero(
        np.array(results[0]["is_detected_v1"])
    ) / count_person[1]

    multi_person_precision = (np.count_nonzero(np.array(results[0]["is_detected_v0"]))
    + np.count_nonzero(np.array(results[0]["is_detected_v1"]))) / np.sum(count_person)

    single_v0_person_distance_error = np.sum(
        np.array(results[0]["single_distance_error_v0"]) * np.array(results[0]["is_detected_v0"])
    ) / np.sum(results[0]["is_detected_v0"]) * single_v0_person_recall + 0.9522 * (1-single_v0_person_recall)
    single_v1_person_distance_error = np.sum(
        np.array(results[0]["single_distance_error_v1"]) * np.array(results[0]["is_detected_v1"])
    ) / np.sum(results[0]["is_detected_v1"]) * single_v1_person_recall + 0.9522 * (1-single_v1_person_recall)

    single_v0_person_az_error = np.sum(
        np.array(results[0]["single_az_error_v0"]) * np.array(results[0]["is_detected_v0"])
    ) / np.sum(results[0]["is_detected_v0"]) * single_v0_person_recall + 5.6 * (1-single_v0_person_recall)
    single_v1_person_az_error = np.sum(
        np.array(results[0]["single_az_error_v1"]) * np.array(results[0]["is_detected_v1"])
    ) / np.sum(results[0]["is_detected_v1"]) * single_v1_person_recall + 5.6 * (1-single_v1_person_recall)

    single_v0_person_el_error = np.sum(
        np.array(results[0]["single_el_error_v0"]) * np.array(results[0]["is_detected_v0"])
    ) / np.sum(results[0]["is_detected_v0"]) * single_v0_person_recall + 5.6 * (1-single_v0_person_recall)
    single_v1_person_el_error = np.sum(
        np.array(results[0]["single_el_error_v1"]) * np.array(results[0]["is_detected_v1"])
    ) / np.sum(results[0]["is_detected_v1"]) * single_v1_person_recall + 5.6 * (1-single_v1_person_recall)

    detected = np.logical_or(
                    np.array(results[0]["is_detected_v0"]), np.array(results[0]["is_detected_v1"])
    )
    multi_person_distance_error = np.sum(np.array(results[0]["multi_distance_error"]) * detected) / np.count_nonzero(detected) * multi_person_recall + 0.9522 * (1-multi_person_recall)
    multi_person_az_error = np.sum(np.array(results[0]["multi_az_error"]) * detected) / np.count_nonzero(detected) * multi_person_recall + 5.6 * (1-multi_person_recall)
    multi_person_el_error = np.sum(np.array(results[0]["multi_el_error"]) * detected) / np.count_nonzero(detected) * multi_person_recall + 5.6 * (1-multi_person_recall)

    results_person_metrices = {
        "single_v0_person_cartesian_coordinate_error [m]": single_v0_person_distance_error,
        "single_v1_person_cartesian_coordinate_error [m]": single_v1_person_distance_error,
        "single_v0_person_az_error [deg]": single_v0_person_az_error,
        "single_v1_person_az_error [deg]": single_v1_person_az_error,
        "single_v0_person_el_error [deg]": single_v0_person_el_error,
        "single_v1_person_el_error [deg]": single_v1_person_el_error,
        "single_v0_person_precision": single_v0_person_precision,
        "single_v1_person_precision": single_v1_person_precision,
        "multi_person_cartesian_coordinate_error [m]": multi_person_distance_error,
        "multi_person_az_error [deg]": multi_person_az_error,
        "multi_person_el_error [deg]": multi_person_el_error,
        "multi_person_recall": multi_person_recall,
        "multi_person_precision": multi_person_precision
    }
    # pprint(results_person_metrices)

    # phone results
    single_v0_phone_recall = np.sum(results[1]["is_detected_v0"]) / len(results[1]["is_available_v0"])
    single_v1_phone_recall = np.sum(results[1]["is_detected_v1"]) / len(results[1]["is_available_v1"])

    multi_single_v0_phone_recall = np.sum(results[1]["is_detected_v0"]) / len(results[1]["is_available_v0"])
    multi_single_v1_phone_recall = np.sum(results[1]["is_detected_v1"]) / len(results[1]["is_available_v1"])

    multi_phone_recall = np.count_nonzero(
        np.array(results[1]["is_detected_v0"]) + np.array(results[1]["is_detected_v1"])
    ) / len(results[1]["is_available_v1"])

    single_v0_phone_precision = np.count_nonzero(
        np.array(results[1]["is_detected_v0"])
    ) / count_phone[0]
    single_v1_phone_precision = np.count_nonzero(
        np.array(results[1]["is_detected_v1"])
    ) / count_phone[1]

    multi_phone_precision = (np.count_nonzero(np.array(results[1]["is_detected_v0"]))
    + np.count_nonzero(np.array(results[1]["is_detected_v1"]))) / np.sum(count_phone)

    single_v0_phone_distance_error = np.sum(
            np.array(results[1]["single_distance_error_v0"]) * np.array(results[1]["is_detected_v0"])
    ) / np.sum(results[1]["is_detected_v0"]) * single_v0_phone_recall + 0.9522 * (1-single_v0_phone_recall)
    single_v1_phone_distance_error = np.sum(
        np.array(results[1]["single_distance_error_v1"]) * np.array(results[1]["is_detected_v1"])
    ) / np.sum(results[1]["is_detected_v1"]) * single_v1_phone_recall + 0.9522 * (1-single_v1_phone_recall)

    single_v0_phone_az_error = np.sum(
        np.array(results[1]["single_az_error_v0"]) * np.array(results[1]["is_detected_v0"])
    ) / np.sum(results[1]["is_detected_v0"]) * single_v0_phone_recall + 5.6 * (1-single_v0_phone_recall)
    single_v1_phone_az_error = np.sum(
        np.array(results[1]["single_az_error_v1"]) * np.array(results[1]["is_detected_v1"])
    ) / np.sum(results[1]["is_detected_v1"]) * single_v1_phone_recall + 5.6 * (1-single_v1_phone_recall)

    single_v0_phone_el_error = np.sum(
        np.array(results[1]["single_el_error_v0"]) * np.array(results[1]["is_detected_v0"])
    ) / np.sum(results[1]["is_detected_v0"]) * single_v0_phone_recall + 5.6 * (1-single_v0_phone_recall)
    single_v1_phone_el_error = np.sum(
        np.array(results[1]["single_el_error_v1"]) * np.array(results[1]["is_detected_v1"])
    ) / np.sum(results[1]["is_detected_v1"]) * single_v1_phone_recall + 5.6 * (1-single_v1_phone_recall)

    detected = np.logical_or(
                    np.array(results[1]["is_detected_v0"]), np.array(results[1]["is_detected_v1"])
    )
    multi_phone_distance_error = np.sum(np.array(results[1]["multi_distance_error"]) * detected) / np.count_nonzero(detected) * multi_phone_recall + 0.9522 * (1-multi_phone_recall)
    multi_phone_az_error = np.sum(np.array(results[1]["multi_az_error"]) * detected) / np.count_nonzero(detected) * multi_phone_recall + 5.6 * (1-multi_phone_recall)
    multi_phone_el_error = np.sum(np.array(results[1]["multi_el_error"]) * detected) / np.count_nonzero(detected) * multi_phone_recall + 5.6 * (1-multi_phone_recall)

    results_phone_metrices = {
        "single_v0_phone_cartesian_coordinate_error [m]": single_v0_phone_distance_error,
        "single_v1_phone_cartesian_coordinate_error [m]": single_v1_phone_distance_error,
        "single_v0_phone_az_error [deg]": single_v0_phone_az_error,
        "single_v1_phone_az_error [deg]": single_v1_phone_az_error,
        "single_v0_phone_el_error [deg]": single_v0_phone_el_error,
        "single_v1_phone_el_error [deg]": single_v1_phone_el_error,
        "single_v0_phone_precision": single_v0_phone_precision,
        "single_v1_phone_precision": single_v1_phone_precision,
        "multi_phone_cartesian_coordinate_error [m]": multi_phone_distance_error,
        "multi_phone_az_error [deg]": multi_phone_az_error,
        "multi_phone_el_error [deg]": multi_phone_el_error,
        "multi_phone_recall": multi_phone_recall,
        "multi_phone_precision": multi_phone_precision
    }
    # pprint(results_phone_metrices)


    single_person_precision = (single_v0_person_precision + single_v1_person_precision) / 2
    single_person_recall = (single_v0_person_recall + single_v1_person_recall) / 2
    single_phone_precision = (single_v0_phone_precision + single_v1_phone_precision) / 2
    single_phone_recall = (single_v0_phone_recall + single_v1_phone_recall) / 2
    single_phone_distance_error = (single_v0_phone_distance_error + single_v1_phone_distance_error) / 2
    single_phone_az_error = (single_v0_phone_az_error + single_v1_phone_az_error) / 2
    single_phone_el_error = (single_v0_phone_el_error + single_v1_phone_el_error) / 2


    result_table = [["Multi-view SVWC (DETR)", f"{multi_person_precision*100:.2f}/{multi_person_recall*100:.2f}", f"{multi_phone_precision*100:.2f}/{multi_phone_recall*100:.2f}", f"{multi_phone_distance_error*100:.2f}", f"{multi_phone_az_error:.2f}/{multi_phone_el_error:.2f}"],
                    ["Single-view SVWC (DETR)", f"{single_person_precision*100:.2f}/{single_person_recall*100:.2f}", f"{single_phone_precision*100:.2f}/{single_phone_recall*100:.2f}", f"{single_phone_distance_error*100:.2f}", f"{(single_v0_phone_az_error+single_v1_phone_az_error)/2:.2f}/{(single_v0_phone_el_error+single_v1_phone_el_error)/2:.2f}"]]


    path_view0 = "./data/gs_data_final/view0/"
    path_view1 = "./data/gs_data_final/view1/"

    place = json.load(open("./data/adjusted_measurement_angles_v3.json"))

    degree_file = "./EfficientDet/vw_kh/degree.npy"
    h, w = 480, 640

    box_file = "./EfficientDet/multiview_measurement.json"
    gt_file = "./EfficientDet/multiview_measurement.json"

    box = json.load(open(box_file))
    box_gt = json.load(open(gt_file))

    place_to_save = copy.deepcopy(place)
    min_distance_to_save = {}
    results = []
    results.append({
        "single_az_error_v0"   :[],
        "single_el_error_v0"   :[],
        "single_az_error_v1"   :[],
        "single_el_error_v1"   :[],
        "single_distance_error_v0":[],
        "single_distance_error_v1":[],
        "is_available_v0"   :[],
        "is_available_v1"   :[],
        "is_detected_v0"    :[],
        "is_detected_v1"    :[],
        "multi_el_error" :[],
        "multi_az_error" :[],
        "multi_distance_error"    :[],
    })

    results.append({
        "single_az_error_v0"   :[],
        "single_el_error_v0"   :[],
        "single_az_error_v1"   :[],
        "single_el_error_v1"   :[],
        "single_distance_error_v0":[],
        "single_distance_error_v1":[],
        "is_available_v0"   :[],
        "is_available_v1"   :[],
        "is_detected_v0"    :[],
        "is_detected_v1"    :[],
        "multi_el_error" :[],
        "multi_az_error" :[],
        "multi_distance_error"    :[],
    })
    detected_person = 0
    detected_phone = 0
    for pp in range(2):
        for p in tqdm(place.keys(), desc=f'Calc Metric for : ', leave=False):
            files_view0 = glob.glob(path_view0 + p + '/' + '*.json')
            files_view1 = glob.glob(path_view1 + p + '/' + '*.json')

            files_view0.sort()
            files_view1.sort()

            View = MultiViewCoordinate(**place[p], degree_file0 = degree_file)

            for f_v0 in files_view0:
                if "convert" in f_v0:
                    continue
                file_index = f_v0.split("_")[-1].split(".")[0]

                d_v0 = json.load(open(f_v0))
                depth_v0 = np.array(d_v0)

                f_v1 = path_view1 + p + '/' + p + '_' + file_index + '.json'
                d_v1 = json.load(open(f_v1))
                depth_v1 = np.array(d_v1)

                # Box info (GT and PD)
                b = box[f"{p}_{file_index}"]
                b_gt = box_gt[f"{p}_{file_index}"]

                b_v0_gt = []
                b_v1_gt = []

                b_v0_pd = []
                b_v1_pd = []

                obj_detected_v0 = []
                obj_detected_v1 = []

                obj_available_v0 = []
                obj_available_v1 = []

                # Object info (GT and PD)
                for k in b_gt.keys():
                    if (pp == 0 and k[:-1] == 'person') or (pp == 1 and k[:-1] == 'phone'):
                        if k not in b.keys():
                            continue
                        # ground truth
                        b_v0_gt.append(
                            b_gt[k]["view0"]["gt_y_x_center"]
                        )
                        b_v1_gt.append(
                            b_gt[k]["view1"]["gt_y_x_center"]
                        )

                        # prediction
                        b_v0_pd.append(
                            b[k]["view0"]["y_x_center"]
                        )
                        b_v1_pd.append(
                            b[k]["view1"]["y_x_center"]
                        )

                        obj_detected_v0.append(b[k]["view0"]["is_detected"])
                        obj_detected_v1.append(b[k]["view1"]["is_detected"])

                        obj_available_v0.append(b[k]["view0"]["is_available"])
                        obj_available_v1.append(b[k]["view1"]["is_available"])

                # is box detected & available in the view
                results[pp]["is_detected_v0"] += obj_detected_v0
                results[pp]["is_detected_v1"] += obj_detected_v1

                results[pp]["is_available_v0"] += obj_available_v0
                results[pp]["is_available_v1"] += obj_available_v1

                xyz_gt_0, xyz_gt_1 = View.to_shared_coordinates(
                    np.array(b_v0_gt),
                    np.array(b_v1_gt),
                    depth_v0,
                    depth_v1
                )

                xyz_pd_0, xyz_pd_1 = View.to_shared_coordinates(
                    np.array(b_v0_pd),
                    np.array(b_v1_pd),
                    depth_v0,
                    depth_v1
                )

                b_v0_gt = np.array(b_v0_gt)
                b_v1_gt = np.array(b_v1_gt)

                b_v0_pd = np.array(b_v0_pd)
                b_v1_pd = np.array(b_v1_pd)

                single_angle_gt0 = View.cam0.to_angle(np.rint(b_v0_gt * np.array([[480,640]])).astype(int))
                single_angle_gt1 = View.cam1.to_angle(np.rint(b_v1_gt * np.array([[480,640]])).astype(int))

                single_angle_pd0 = View.cam0.to_angle(np.rint(b_v0_pd * np.array([[480,640]])).astype(int))
                single_angle_pd1 = View.cam1.to_angle(np.rint(b_v1_pd * np.array([[480,640]])).astype(int))

                single_angle_error0 = angle_error(np.array(single_angle_gt0).transpose(),
                                                np.array(single_angle_pd0).transpose())
                single_angle_error1 = angle_error(np.array(single_angle_gt1).transpose(),
                                                np.array(single_angle_pd1).transpose())

                single_distance_error0 = distance_error(xyz_gt_0, xyz_pd_0)
                single_distance_error1 = distance_error(xyz_gt_1, xyz_pd_1)

                m_a_s0 = single_angle_error0.copy()
                m_a_s1 = single_angle_error1.copy()

                m_a_s0[np.array(obj_detected_v0)==0,:] = 1e5
                m_a_s1[np.array(obj_detected_v1)==0,:] = 1e5

                m_d_s0 = single_distance_error0.copy()
                m_d_s1 = single_distance_error1.copy()

                m_d_s0[np.array(obj_detected_v0)==0] = 1e5
                m_d_s1[np.array(obj_detected_v1)==0] = 1e5

                multi_el_error = np.min([
                    m_a_s0[:,0],
                    m_a_s1[:,0]
                ], 0).flatten().tolist()
                multi_az_error = np.min([
                    m_a_s0[:,1],
                    m_a_s1[:,1]
                ], 0).flatten().tolist()
                multi_distance_error = np.min([
                    m_d_s0,
                    m_d_s1
                ], 0).flatten().tolist()

                # Single-view
                results[pp]["single_el_error_v0"] = results[pp]["single_el_error_v0"] + single_angle_error0[:,0].flatten().tolist()
                results[pp]["single_el_error_v1"] = results[pp]["single_el_error_v1"] + single_angle_error1[:,0].flatten().tolist()

                results[pp]["single_az_error_v0"] = results[pp]["single_az_error_v0"] + single_angle_error0[:,1].flatten().tolist()
                results[pp]["single_az_error_v1"] = results[pp]["single_az_error_v1"] + single_angle_error1[:,1].flatten().tolist()

                results[pp]["single_distance_error_v0"] = results[pp]["single_distance_error_v0"] + single_distance_error0.flatten().tolist()
                results[pp]["single_distance_error_v1"] = results[pp]["single_distance_error_v1"] + single_distance_error1.flatten().tolist()

                # Multi-view
                results[pp]["multi_el_error"] = results[pp]["multi_el_error"] + multi_el_error
                results[pp]["multi_az_error"] = results[pp]["multi_az_error"] + multi_az_error
                results[pp]["multi_distance_error"] = results[pp]["multi_distance_error"] + multi_distance_error


    count_person = np.array(box["num_person"])
    count_phone = np.array(box["num_phone"])

    # person results
    single_v0_person_recall = np.sum(results[0]["is_detected_v0"]) / len(results[0]["is_available_v0"])
    single_v1_person_recall = np.sum(results[0]["is_detected_v1"]) / len(results[0]["is_available_v1"])

    single_v0_person_precision = np.count_nonzero(
        np.array(results[0]["is_detected_v0"])
    ) / count_person[0]
    single_v1_person_precision = np.count_nonzero(
        np.array(results[0]["is_detected_v1"])
    ) / count_person[1]

    single_v0_person_distance_error = np.sum(
        np.array(results[0]["single_distance_error_v0"]) * np.array(results[0]["is_detected_v0"])
    ) / np.sum(results[0]["is_detected_v0"]) * single_v0_person_recall + 0.9522 * (1-single_v0_person_recall)
    single_v1_person_distance_error = np.sum(
        np.array(results[0]["single_distance_error_v1"]) * np.array(results[0]["is_detected_v1"])
    ) / np.sum(results[0]["is_detected_v1"]) * single_v1_person_recall + 0.9522 * (1-single_v1_person_recall)

    single_v0_person_az_error = np.sum(
        np.array(results[0]["single_az_error_v0"]) * np.array(results[0]["is_detected_v0"])
    ) / np.sum(results[0]["is_detected_v0"]) * single_v0_person_recall + 5.6 * (1-single_v0_person_recall)
    single_v1_person_az_error = np.sum(
        np.array(results[0]["single_az_error_v1"]) * np.array(results[0]["is_detected_v1"])
    ) / np.sum(results[0]["is_detected_v1"]) * single_v1_person_recall + 5.6 * (1-single_v1_person_recall)

    single_v0_person_el_error = np.sum(
        np.array(results[0]["single_el_error_v0"]) * np.array(results[0]["is_detected_v0"])
    ) / np.sum(results[0]["is_detected_v0"]) * single_v0_person_recall + 5.6 * (1-single_v0_person_recall)
    single_v1_person_el_error = np.sum(
        np.array(results[0]["single_el_error_v1"]) * np.array(results[0]["is_detected_v1"])
    ) / np.sum(results[0]["is_detected_v1"]) * single_v1_person_recall + 5.6 * (1-single_v1_person_recall)

    detected = np.logical_or(
                    np.array(results[0]["is_detected_v0"]), np.array(results[0]["is_detected_v1"])
    )

    results_person_metrices = {
        "single_v0_person_cartesian_coordinate_error [m]": single_v0_person_distance_error,
        "single_v1_person_cartesian_coordinate_error [m]": single_v1_person_distance_error,
        "single_v0_person_az_error [deg]": single_v0_person_az_error,
        "single_v1_person_az_error [deg]": single_v1_person_az_error,
        "single_v0_person_el_error [deg]": single_v0_person_el_error,
        "single_v1_person_el_error [deg]": single_v1_person_el_error,
        "single_v0_person_precision": single_v0_person_precision,
        "single_v1_person_precision": single_v1_person_precision,
    }
    # pprint(results_person_metrices)

    # phone results
    single_v0_phone_recall = np.sum(results[1]["is_detected_v0"]) / len(results[1]["is_available_v0"])
    single_v1_phone_recall = np.sum(results[1]["is_detected_v1"]) / len(results[1]["is_available_v1"])

    multi_single_v0_phone_recall = np.sum(results[1]["is_detected_v0"]) / len(results[1]["is_available_v0"])
    multi_single_v1_phone_recall = np.sum(results[1]["is_detected_v1"]) / len(results[1]["is_available_v1"])

    single_v0_phone_precision = np.count_nonzero(
        np.array(results[1]["is_detected_v0"])
    ) / count_phone[0]
    single_v1_phone_precision = np.count_nonzero(
        np.array(results[1]["is_detected_v1"])
    ) / count_phone[1]

    single_v0_phone_distance_error = np.sum(
            np.array(results[1]["single_distance_error_v0"]) * np.array(results[1]["is_detected_v0"])
    ) / np.sum(results[1]["is_detected_v0"]) * single_v0_phone_recall + 0.9522 * (1-single_v0_phone_recall)
    single_v1_phone_distance_error = np.sum(
        np.array(results[1]["single_distance_error_v1"]) * np.array(results[1]["is_detected_v1"])
    ) / np.sum(results[1]["is_detected_v1"]) * single_v1_phone_recall + 0.9522 * (1-single_v1_phone_recall)

    single_v0_phone_az_error = np.sum(
        np.array(results[1]["single_az_error_v0"]) * np.array(results[1]["is_detected_v0"])
    ) / np.sum(results[1]["is_detected_v0"]) * single_v0_phone_recall + 5.6 * (1-single_v0_phone_recall)
    single_v1_phone_az_error = np.sum(
        np.array(results[1]["single_az_error_v1"]) * np.array(results[1]["is_detected_v1"])
    ) / np.sum(results[1]["is_detected_v1"]) * single_v1_phone_recall + 5.6 * (1-single_v1_phone_recall)

    single_v0_phone_el_error = np.sum(
        np.array(results[1]["single_el_error_v0"]) * np.array(results[1]["is_detected_v0"])
    ) / np.sum(results[1]["is_detected_v0"]) * single_v0_phone_recall + 5.6 * (1-single_v0_phone_recall)
    single_v1_phone_el_error = np.sum(
        np.array(results[1]["single_el_error_v1"]) * np.array(results[1]["is_detected_v1"])
    ) / np.sum(results[1]["is_detected_v1"]) * single_v1_phone_recall + 5.6 * (1-single_v1_phone_recall)

    detected = np.logical_or(
                    np.array(results[1]["is_detected_v0"]), np.array(results[1]["is_detected_v1"])
    )

    results_phone_metrices = {
        "single_v0_phone_cartesian_coordinate_error [m]": single_v0_phone_distance_error,
        "single_v1_phone_cartesian_coordinate_error [m]": single_v1_phone_distance_error,
        "single_v0_phone_az_error [deg]": single_v0_phone_az_error,
        "single_v1_phone_az_error [deg]": single_v1_phone_az_error,
        "single_v0_phone_el_error [deg]": single_v0_phone_el_error,
        "single_v1_phone_el_error [deg]": single_v1_phone_el_error,
        "single_v0_phone_precision": single_v0_phone_precision,
        "single_v1_phone_precision": single_v1_phone_precision,
    }
    # pprint(results_phone_metrices)


    single_person_precision = (single_v0_person_precision + single_v1_person_precision) / 2
    single_person_recall = (single_v0_person_recall + single_v1_person_recall) / 2
    single_phone_precision = (single_v0_phone_precision + single_v1_phone_precision) / 2
    single_phone_recall = (single_v0_phone_recall + single_v1_phone_recall) / 2
    single_phone_distance_error = (single_v0_phone_distance_error + single_v1_phone_distance_error) / 2
    single_phone_az_error = (single_v0_phone_az_error + single_v1_phone_az_error) / 2
    single_phone_el_error = (single_v0_phone_el_error + single_v1_phone_el_error) / 2

    result_table.extend([["Single-view SVWC (EfficientDet)", f"{single_person_precision*100:.2f}/{single_person_recall*100:.2f}", f"{single_phone_precision*100:.2f}/{single_phone_recall*100:.2f}", f"{single_phone_distance_error*100:.2f}", f"{single_phone_az_error:.2f}/{single_phone_el_error:.2f}"]])

    from tabulate import tabulate
    print(tabulate(result_table, headers=["Technique","Human precision/recall (%)", "Cell phone precision/recall (%)", "distance error (cm)", "az/el angle error (degree)"], tablefmt="github", numalign="left"))

