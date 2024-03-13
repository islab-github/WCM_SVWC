import json
import argparse
import copy

def sum_json(args:dict):
    data = json.load(open(args["File_path"] + '/' + args["File_name"] + '_convert.json'))
    data2 = json.load(open(args["File_path2"] + '/' + args["File_name"] + '_convert.json'))

    final_data = copy.deepcopy(data)
    if "results" in args.keys():
        file_path = args["results"] + '/' + args["File_name"]

    for p in data.keys():
        i = p.split("_")[-1]
        object = list(set(list(data[p].keys()) + list(data2[p].keys())))
        for j in range(len(object)):
            if object[j] in data[args["File_name"] + f'_{i}'].keys():
                data[args["File_name"] + f'_{i}'][object[j]]['view0']["is_available"] = 1.0
                final_data[args["File_name"] + f'_{i}'][object[j]].update(data[args["File_name"] + f'_{i}'][object[j]])
                if object[j] in data2[args["File_name"] + f'_{i}'].keys():
                    data2[args["File_name"] + f'_{i}'][object[j]]['view1']["is_available"] = 1.0
                    if object[j] in final_data[args["File_name"] + f'_{i}'].keys():
                        final_data[args["File_name"] + f'_{i}'][object[j]].update(data2[args["File_name"] + f'_{i}'][object[j]])
                    else:
                        final_data[args["File_name"] + f'_{i}'][object[j]] = data2[args["File_name"] + f'_{i}'][object[j]]
                else:
                    temp_column_dict = {
                        "view1": {
                            "y_x_center": [0.0, 0.0],
                            "gt_y_x_center": [0.0, 0.0],
                            "width": 0.0,
                            "length": 0.0,
                            "is_detected": 0.0,
                            "is_available": 0.0
                        }
                    }
                    final_data[args["File_name"] + f'_{i}'][object[j]].update(temp_column_dict)

            else:
                if object[j] in data2[args["File_name"] + f'_{i}'].keys():
                    data2[args["File_name"] + f'_{i}'][object[j]]['view1']["is_available"] = 1.0
                    if object[j] in final_data[args["File_name"] + f'_{i}'].keys():
                        final_data[args["File_name"] + f'_{i}'][object[j]].update(data2[args["File_name"] + f'_{i}'][object[j]])
                    else:
                        final_data[args["File_name"] + f'_{i}'][object[j]] = data2[args["File_name"] + f'_{i}'][object[j]]
                    temp_column_dict = {
                        "view0": {
                            "y_x_center": [0.0, 0.0],
                            "gt_y_x_center": [0.0, 0.0],
                            "width": 0.0,
                            "length": 0.0,
                            "is_detected": 0.0,
                            "is_available": 0.0
                        }
                    }
                    final_data[args["File_name"] + f'_{i}'][object[j]].update(temp_column_dict)
                else:
                    temp_column_dict = {
                        "view1": {
                            "y_x_center": [0.0, 0.0],
                            "gt_y_x_center": [0.0, 0.0],
                            "width": 0.0,
                            "length": 0.0,
                            "is_detected": 0.0,
                            "is_available": 0.0
                        }
                    }
                    final_data[args["File_name"] + f'_{i}'][object[j]].update(temp_column_dict)

    if "results" not in args.keys():
        return final_data

    else:
        with open(file_path + '/' + args["File_name"] + '.json', 'w') as f:
            json.dump(final_data, f, indent=4)
        print(file_path + '/' + args["File_name"] + '/' + args["File_name"] + '.json')
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="path of data and file name")
    parser.add_argument('--File_path', help='file path')
    parser.add_argument('--File_path2', help='file name')
    parser.add_argument('--File_name', help='file name')
    parser.add_argument('--results', help='file name')
    args = parser.parse_args()

    args = vars(args)
    sum_json(args)
