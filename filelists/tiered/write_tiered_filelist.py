import os
from os import listdir
from os.path import isfile, isdir, join
import json
import random
import numpy as np

# cwd = os.getcwd()
TIRED_IMAGENET_DLP = "/input/kaggle/tiered_imagenet"
DEFAULT_SAVE_DIR = "./"

data_path = TIRED_IMAGENET_DLP
savedir = DEFAULT_SAVE_DIR
dataset_list = ["base", "val", "novel"]

for dataset in dataset_list:
    # get folder list for each dataset
    if dataset == "base":
        split = "train"
    elif dataset == "val":
        split = "val"
    elif dataset == "novel":
        split = "test"
    split_path = os.path.join(data_path, split)
    folder_list = listdir(split_path)

    
    # get file and label list
    file_list = []
    label_list = []
    for folder_id, folder in enumerate(folder_list):
        folder_path = os.path.join(split_path, folder)
        imgs = listdir(folder_path)
        for img in imgs:
            file_list.append(os.path.join(folder_path, img)) # add path_2_file list
            label_list.append(folder_id) # encode label by folder id

    # save filelist
    filelist = {
        "label_names": folder_list,
        "image_names": file_list,
        "image_labels": label_list
    }
    with open(savedir + dataset + ".json", "w") as fo:
        json.dump(filelist, fo, indent=4)
        fo.close()
    print("%s -OK" % dataset)
