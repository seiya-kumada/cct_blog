#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import glob


def to_image(x):
    x = x.view(x.size(0), 3, 224, 224)
    return x


def load_list(list_path):
    lines = []
    for line in open(list_path):
        if ".jpg" in line:
            line = line.strip()
            lines.append(line)
    return lines


def load_images(dir_path):
    paths = glob.glob(os.path.join(dir_path, "*.jpg"))
    results = []
    for path in paths:
        # if is_ignored(path):
        #     continue
        results.append(path)
    return results
