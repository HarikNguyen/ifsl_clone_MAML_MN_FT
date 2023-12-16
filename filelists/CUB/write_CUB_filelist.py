import os
from os import listdir
from os.path import isfile, isdir, join
import json
import random
import numpy as np

CUB_DLP = "/kaggle/input/cub-200-2011/CUB_200_2011/CUB_200_2011/images"
# cwd = os.getcwd()
# CUB_DLP = join(cwd, "CUB_200_2011/images")
DEFAULT_SAVE_DIR = "./"

data_path = CUB_DLP
savedir = DEFAULT_SAVE_DIR
dataset_list = ["base", "val", "novel"]

# create savedir (if not exist)
if not os.path.exists(savedir) and savedir != DEFAULT_SAVE_DIR:
   os.makedirs(savedir)

# get folder list in dataset
folder_list = [folder for folder in listdir(data_path) if isdir(join(data_path, folder))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

classfile_list_all = []

for folder_id, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append(
        [
            join(folder_path, classfile)
            for classfile in listdir(folder_path)
            if (isfile(join(folder_path, classfile)) and classfile[0] != ".")
        ]
    )
    random.shuffle(classfile_list_all[folder_id])


for dataset in dataset_list:
    file_list = []
    label_list = []
    for classfile_list_id, classfile_list in enumerate(classfile_list_all):
        if "base" in dataset:
            if classfile_list_id % 2 == 0:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(classfile_list_id, len(classfile_list)).tolist()
        if "val" in dataset:
            if classfile_list_id % 4 == 1:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(classfile_list_id, len(classfile_list)).tolist()
        if "novel" in dataset:
            if classfile_list_id % 4 == 3:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(classfile_list_id, len(classfile_list)).tolist()

    filelists = {
        "label_names": folder_list,
        "image_names": file_list,
        "image_labels": label_list
    }

    with open(savedir + dataset + ".json", "w") as fo:
        json.dump(filelists, fo, indent=4)
        fo.close()
    print("%s -OK" % dataset)
