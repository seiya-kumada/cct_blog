#!/usr/bin/env python
# -*- conding:utf-8 -*-
import pickle
import sklearn.manifold as manifold
import matplotlib.pyplot as plt

FEATURE_PATH = "/Users/kumada/Downloads/fs.pkl"


def store(l, fs, rls, rfs):
    for f in fs:
        rls.append(l)
        rfs.append(f)


if __name__ == "__main__":
    data = pickle.load(open(FEATURE_PATH, "rb"))
    rlabels = []
    rfeatures = []
    for (label, features) in data.items():
        store(label, features, rlabels, rfeatures)

    reduced_data = manifold.TSNE(n_components=2, random_state=0).fit_transform(rfeatures)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=rlabels, marker='.')
    plt.colorbar()

    plt.savefig("./distribution.jpg")
    # plt.show()
