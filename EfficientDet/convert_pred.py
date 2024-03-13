import os
import json
import numpy as np
import argparse
def convert(x1, y1, x2, y2):
    center_x = (x1 + x2)/2.0
    center_y = (y1 + y2)/2.0
    width = x2-x1
    length = y2-y1
    center_x = center_x / 640
    center_y = center_y / 480
    width = width / 640
    length = length / 480
    return center_x, center_y, width, length

def abs_diff(gt_x, x, gt_y, y):
    return abs(gt_x-x) + abs(gt_y-y)


def convert_pred(args:dict):

    our_cls = []

    with open(args["File_path"] + '/' + "classes.txt", "r", encoding="utf-8") as cls:
        for cl in cls.readlines():
            cl = cl.replace("\n", "")
            our_cls.append(cl)
    result = {}
    data = json.load(open(args["result_file"], "r", encoding="utf-8"))
    for i in range(0, 15):
        if os.path.exists(args["File_path"] + '/' + args["File_name"] + f'_{i}.txt'):
            file_name = args["File_name"] + f'_{i}.png'
            file_data = data[file_name]
            result[args["File_name"] + f'_{i}'] = {}
            with open(args["File_path"] + '/' + args["File_name"] + f'_{i}.txt', "r", encoding="utf-8") as txt:
                for line in txt.readlines():
                    min = 9999999
                    newline = line.replace("\n", "")
                    name = newline.split(" ")
                    for j in range(len(file_data)):
                        idx = list(file_data.keys())[j]
                        score = file_data[idx]['score']

                        if ("person" in file_data[idx]["obj"] and score <= args["threshold_person"]) or \
                            ("phone" in file_data[idx]["obj"] and score <= args["threshold_phone"]):
                            continue

                        x1 = file_data[idx]['x1']
                        x2 = file_data[idx]['x2']
                        y1 = file_data[idx]['y1']
                        y2 = file_data[idx]['y2']

                        x_center, y_center, width, length = convert(x1, y1, x2, y2)
                        ans = abs_diff(float(name[1]), x_center, float(name[2]), y_center) #, float(name[3]), width, float(name[4]), length)
                        if ans< min:
                            min = ans
                            t = j

                    if (min > 0.045 and "person" in our_cls[int(name[0])]) or \
                            (min > 0.03 and "phone" in our_cls[int(name[0])]):
                        x_center = 0.0
                        y_center = 0.0
                        width = 0.0
                        length = 0.0
                        obj = our_cls[int(name[0])]
                        is_dect = 0.0

                    else:
                        x1 = file_data[list(file_data.keys())[t]]['x1']
                        x2 = file_data[list(file_data.keys())[t]]['x2']
                        y1 = file_data[list(file_data.keys())[t]]['y1']
                        y2 = file_data[list(file_data.keys())[t]]['y2']
                        obj = our_cls[int(name[0])]
                        x_center, y_center, width, length = convert(x1, y1, x2, y2)
                        is_dect = 1.0

                    temp_column_dict = {
                                f"view{args['view']}":{
                                    "y_x_center": [round(y_center, 4), round(x_center, 4)],
                                    "gt_y_x_center": [round(float(name[2]), 4), round(float(name[1]), 4)],
                                    "width": width,
                                    "length": length,
                                    "gt_width": round(float(name[3]),4),
                                    "gt_length": round(float(name[4]), 4),
                                    "is_detected": is_dect
                                }
                            }


                    result[args["File_name"] + f'_{i}'][obj] = temp_column_dict

    with open(args["File_path"] + '/' + args["File_name"] + '_convert.json', 'w') as f:
        json.dump(result, f, indent=4)
    print(args["File_path"] + '/' + args["File_name"] + '_convert.json')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="path of data and file name")
#     parser.add_argument('--File_path', help='file path')
#     parser.add_argument('--File_name', help='file name')
#     parser.add_argument('--result_file', help='result file name')
#     parser.add_argument('--view', help='result file name')
#     args = parser.parse_args()

#     args = vars(args)
#     convert_pred(args)


