from collections import OrderedDict
import os
import json

 
def load_data_view_wj(DATA_DIR):
    data = OrderedDict()
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue

            if file == "classes.txt":
                continue

            if "crop" in root:
                continue

            # gt_file ends with ".txt"
            directory = root.split("\\")[-1]
            gt_path = os.path.join(root, file)
            rgb_path = gt_path.replace(".txt", ".png")
            distance_path = gt_path.replace(".txt", ".json")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".txt", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path, directory, file)

    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data



def load_prediction(json_path: str = "data/prediction.json"):
    with open(json_path, "r") as f:
        pred = json.load(f)

    pred = OrderedDict(sorted(pred.items()))
    print(f"Prediction loaded, total: {len(pred)} predictions")
    return pred


if __name__ == '__main__':
    p = load_prediction()
    d = load_data_view_wj()

    assert len(p) == len(d)

    # keys are different, but anyway they are sorted...

    # for (pk, dk) in zip(p.keys(), d.keys()):
    #     pk = pk.replace(".png", "")
    #     dk = dk.replace(".png", "")
    #     if dk not in pk:
    #         print(pk, dk)
