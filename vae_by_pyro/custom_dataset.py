#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from PIL import Image
import torchvision.transforms as transforms
# import utils
# import conf
# import custom_dataset as cd
import os


def normalize_with_minmax(x, min_value, max_value):
    min_x = x.min()
    max_x = x.max()
    x = (x - min_x) / (max_x - min_x)
    x = x * (max_value - min_value) + min_value
    return x


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, image_size, paths):
        self.paths = paths
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: normalize_with_minmax(x, 0, 1)),
            # transforms.Lambda(lambda x: torch.round(x)),

        ])

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        x = self.preprocess(image)
        _, tail = os.path.split(path)
        return x, tail

    def __len__(self):
        return len(self.paths)


# if __name__ == "__main__":
    # paths = utils.load_images(conf.INPUT_DIR_PATH)
    # custom_dataset = cd.CustomDataset(conf.IMAGE_SIZE, paths)
    # for x, path in custom_dataset:
    #     print(x.size(), path)
    #     break
