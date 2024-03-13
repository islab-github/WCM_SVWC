from collections import OrderedDict
import os
import json

LABEL_DIR = "../../../../Downloads/VisionWireless/data/label"
DATA_DIR = "../../../../Downloads/VisionWireless/data/result"


def load_data():
    data = OrderedDict()  # key: (gt_path, rgb_path, distance_path)

    # indoor
    for root, dirs, files in os.walk(f"{LABEL_DIR}/indoor"):
        for file in files:
            if not file.endswith(".xml"):
                continue

            gt_path = f"{LABEL_DIR}/indoor/{file}"
            rgb_path = f"{DATA_DIR}/indoor/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/indoor/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path)

    # outdoor
    for root, dirs, files in os.walk(f"{LABEL_DIR}/outdoor"):
        for file in files:
            if not file.endswith(".xml"):
                continue

            gt_path = f"{LABEL_DIR}/outdoor/{file}"
            rgb_path = f"{DATA_DIR}/outdoor/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/outdoor/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = "outdoor_" + file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path)

    # outdoor_v2
    for root, dirs, files in os.walk(f"{LABEL_DIR}/outdoor_v2"):
        for file in files:
            if not file.endswith(".xml"):
                continue

            gt_path = f"{LABEL_DIR}/outdoor_v2/{file}"
            rgb_path = f"{DATA_DIR}/outdoor_v2/rgb/{file}".replace(".xml", ".png")
            distance_path = f"{DATA_DIR}/outdoor_v2/distance/{file}".replace(".xml", ".json")
            distance_path = distance_path.replace("cal", "cal_distance_")
            assert os.path.isfile(gt_path), f"GT {gt_path} does not exist."
            assert os.path.isfile(rgb_path), f"RGB {rgb_path} does not exist."
            assert os.path.isfile(distance_path), f"Distance {distance_path} does not exist."

            rgb_key = file.replace(".xml", ".png")
            data[rgb_key] = (gt_path, rgb_path, distance_path)

    data = OrderedDict(sorted(data.items()))  # noqa
    print(f"Data loaded, total: {len(data)} sets")
    return data


def load_prediction():
    with open("../../../../Downloads/VisionWireless/data/prediction.json", "r") as f:
        pred = json.load(f)

    pred = OrderedDict(sorted(pred.items()))
    print(f"Prediction loaded, total: {len(pred)} predictions")
    return pred


if __name__ == '__main__':
    p = load_prediction()
    d = load_data()

    assert len(p) == len(d)

    # keys are different, but anyway they are sorted...

    # for (pk, dk) in zip(p.keys(), d.keys()):
    #     pk = pk.replace(".png", "")
    #     dk = dk.replace(".png", "")
    #     if dk not in pk:
    #         print(pk, dk)
