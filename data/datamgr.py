# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import (
    SimpleDataset,
    SetDataset,
    EpisodicBatchSampler,
    SimpleTieredDataset,
)
from abc import abstractmethod


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        if transform_type == "RandomSizedCrop":
            method = getattr(transforms, "RandomResizedCrop")
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            method = getattr(transforms, transform_type)
            return method(self.image_size)
        elif transform_type == "Scale":
            method = getattr(transforms, "Resize")
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            method = getattr(transforms, transform_type)
            return method(**self.normalize_param)
        else:
            method = getattr(transforms, transform_type)
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = [
                "RandomSizedCrop",
                "ImageJitter",
                "RandomHorizontalFlip",
                "ToTensor",
                "Normalize",
            ]
        else:
            transform_list = ["Scale", "CenterCrop", "ToTensor", "Normalize"]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(
        self, data_file, aug, num_workers=12, tiered_mini=False
    ):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if not tiered_mini:
            dataset = SimpleDataset(data_file, transform)
        else:
            dataset = SimpleTieredDataset(data_file, transform)
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(
        self, data_file, aug, debug=False
    ):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        n_workers = 12
        if debug:
            n_workers = 0
        data_loader_params = dict(
            batch_sampler=sampler, num_workers=n_workers, pin_memory=True
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
