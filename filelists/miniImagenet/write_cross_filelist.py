import os
import json
import random
import numpy as np
import pandas as pd

# cwd = os.getcwd()
# MINI_IMAGENET_DLP = join(cwd,'ILSVRC2015/Data/CLS-LOC/train')
MINI_IMAGENET_DLP = "/kaggle/input/mini-imagenet/images"
DEFAULT_SAVE_DIR = "./"

data_path = MINI_IMAGENET_DLP
savedir = DEFAULT_SAVE_DIR
dataset_list = ["base", "val", "novel"]

if not os.path.exists(savedir) and savedir != DEFAULT_SAVE_DIR:
    os.makedirs(savedir)

datasetmap = {"base": "train", "val": "val", "novel": "test"}
filelists = {"base": {}, "val": {}, "novel": {}}

past_end_label_code = -1
for dataset in dataset_list:
    # read {dataset}.csv
    dataset_describe_df = pd.read_csv(f"./{datasetmap[dataset]}.csv")
    # get label_names
    label_names = dataset_describe_df.label.unique()

    # encode label_names
    label_names_encoded = list(
        range(past_end_label_code + 1, past_end_label_code + 1 + label_names.shape[0])
    )
    past_end_label_code = past_end_label_code + label_names.shape[0]

    # get path and label encoded for each image
    image_names = []
    image_labels = []
    for label, label_code in zip(label_names.tolist(), label_names_encoded):
        # filter by label
        filenames = dataset_describe_df[dataset_describe_df["label"] == label][
            "filename"
        ]

        # add path
        filenames = data_path + "/" + filenames
        filenames = filenames.tolist()

        # shuffle filenames
        random.shuffle(filenames)

        # add results
        image_names.extend(filenames)
        image_labels.extend(np.repeat(label_code, len(filenames)).tolist())

    filelists[dataset] = {
        "label_names": list(label_names),
        "image_names": list(image_names),
        "image_labels": list(image_labels),
    }

# cross setting use base/val/novel together
filelists_all = {"label_names": [], "image_names": [], "image_labels": []}
for filelist in filelists.values():
    filelists_all["label_names"].extend(filelist["label_names"])
    filelists_all["image_names"].extend(filelist["image_names"])
    filelists_all["image_labels"].extend(filelist["image_labels"])


# write to json file (describe path to dataset)
with open(savedir + "all.json", "w") as fo:
    json.dump(filelists_all, fo, indent=4)
    fo.close()

print("all -OK")
