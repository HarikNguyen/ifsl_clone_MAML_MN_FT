import os

save_dir = "/kaggle/output"  # Change to desired saving dir
data_dir = {}
data_dir["CUB"] = "./filelists/CUB/"
data_dir["miniImagenet"] = "./filelists/miniImagenet/"
data_dir["tiered"] = "./filelists/tiered/"

# path to pretrain weight
simple_shot_dir = "/kaggle/working/"  # Location of the downloaded pretrained model
feat_dir = (
    "/data2/yuezhongqi/Model/feat/"  # Location of the downloaded pretrained model
)
tiered_dir = (
    "/kaggle/input/tiered-imagenet/tiered_imagenet"  # Location of the downloaded tieredImageNet
)
