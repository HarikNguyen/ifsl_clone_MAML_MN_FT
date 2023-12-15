import os
import json
import random
import numpy as np
import pandas as pd

MINI_IMAGENET_DLP = "/input/kaggle/mini-imagenet/images"
DEFAULT_SAVE_DIR = "./"

cwd = os.getcwd()

data_path = MINI_IMAGENET_DLP
savedir = DEFAULT_SAVE_DIR
dataset_list = ["base", "val", "novel"]

# create savedir if it not exist and is not a default save dir
if not os.path.exists(savedir) and savedir != DEFAULT_SAVE_DIR:
    os.makedirs(savedir)

datasetmap = {"base": "train", "val": "val", "novel": "test"}
filelists = {"base": {}, "val": {}, "novel": {}}

# get path to dataset
for dataset in dataset_list:
    # read {dataset}.csv
    dataset_describe_df = pd.read_csv(f"./{datasetmap[dataset]}.csv")

    # get label_names
    label_names = dataset_describe_df.label.unique()

    # encode label_names
    label_names_encoded = list(range(label_names.shape[0]))

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


# write to json file (describe path to dataset)
for dataset in dataset_list:
    with open(savedir + dataset + ".json", "w") as fo:
        json.dump(filelists[dataset], fo)

    print("%s -OK" % dataset)
