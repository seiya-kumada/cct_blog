#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
ROOT_DIR_PATH = "/home/ubuntu/data/text/"
SRC_DIR_PATH = os.path.join(ROOT_DIR_PATH, "kaden-channel")
DST_PATH = os.path.join(ROOT_DIR_PATH, "kaden-channel.txt")


def extract_title(path):
    with open(path) as fin:
        lines = fin.readlines()
        return lines[2].strip()


def extract_titles(src_dir_path, dst_path):
    titles = []
    for name in os.listdir(src_dir_path):
        path = os.path.join(src_dir_path, name)
        title = extract_title(path)
        titles.append(title)

    with open(dst_path, 'w') as fout:
        for title in titles:
            fout.write("{}\n".format(title))


if __name__ == "__main__":
    for (root, dirs, files) in os.walk(ROOT_DIR_PATH):
        for dir in dirs:
            src_dir_path = os.path.join(root, dir)
            dst_path = os.path.join(root, "{}.txt".format(dir))
            extract_titles(src_dir_path, dst_path)
