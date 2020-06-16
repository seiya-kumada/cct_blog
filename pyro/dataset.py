#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.distributions as D
from sklearn.datasets import load_iris
import numpy as np


def make_iris_dataset():
    iris = load_iris()
    data = iris.data
    targets = iris.target
    # names = iris.target_names
    # 0:setosa, 1:versicolor, 2:virginica

    xs = data[:, 2].astype(np.float32)
    ys = data[:, 3].astype(np.float32)
    cs = []
    for t in targets:
        if t == 0:  # setosa
            cs.append("r")
        if t == 1:  # versicolor
            cs.append("g")
        if t == 2:  # virginica
            cs.append("b")
    return torch.tensor(np.stack([xs, ys], axis=1)), cs


def make_dataset_0(obs_num, dim, k):
    centers = torch.tensor([
        [-10.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0]])

    loc_0 = centers[0]
    cov_0 = torch.eye(dim) * 2.0
    dis_0 = D.MultivariateNormal(loc=loc_0, covariance_matrix=cov_0)

    loc_1 = centers[1]
    cov_1 = torch.eye(dim) * 2.0
    dis_1 = D.MultivariateNormal(loc=loc_1, covariance_matrix=cov_1)

    loc_2 = centers[2]
    cov_2 = torch.eye(dim) * 2.0
    dis_2 = D.MultivariateNormal(loc=loc_2, covariance_matrix=cov_2)

    values = []
    for _ in range(obs_num // k):
        a = dis_0.sample()
        b = dis_1.sample()
        c = dis_2.sample()
        values.append(a)
        values.append(b)
        values.append(c)
    return torch.stack(values, dim=0)


def make_dataset_1(obs_num, dim, k):
    centers = torch.tensor([
        [-2.0, -2.0],
        [8.0, 0.0],
        [0.0, 8.0]])

    loc_0 = centers[0]
    cov_0 = torch.tensor([[10.0, 9], [9, 10]])
    dis_0 = D.MultivariateNormal(loc=loc_0, covariance_matrix=cov_0)

    loc_1 = centers[1]
    cov_1 = torch.eye(dim) * 2.0
    dis_1 = D.MultivariateNormal(loc=loc_1, covariance_matrix=cov_1)

    loc_2 = centers[2]
    cov_2 = torch.eye(dim) * 2.0
    dis_2 = D.MultivariateNormal(loc=loc_2, covariance_matrix=cov_2)

    values = []
    for _ in range(obs_num // k):
        a = dis_0.sample()
        b = dis_1.sample()
        c = dis_2.sample()
        values.append(a)
        values.append(b)
        values.append(c)
    return torch.stack(values, dim=0)


def make_d_dataset(obs_num, dim, k):
    centers = 10.0 * torch.eye(dim)

    loc_0 = centers[0]
    cov_0 = torch.eye(dim) * 2.0
    dis_0 = D.MultivariateNormal(loc=loc_0, covariance_matrix=cov_0)

    loc_1 = centers[1]
    cov_1 = torch.eye(dim) * 2.0
    dis_1 = D.MultivariateNormal(loc=loc_1, covariance_matrix=cov_1)

    loc_2 = centers[2]
    cov_2 = torch.eye(dim) * 2.0
    dis_2 = D.MultivariateNormal(loc=loc_2, covariance_matrix=cov_2)

    loc_3 = centers[3]
    cov_3 = torch.eye(dim) * 2.0
    dis_3 = D.MultivariateNormal(loc=loc_3, covariance_matrix=cov_3)

    values = []
    for _ in range(obs_num // k):
        a = dis_0.sample()
        b = dis_1.sample()
        c = dis_2.sample()
        d = dis_3.sample()
        values.append(a)
        values.append(b)
        values.append(c)
        values.append(d)
    return torch.stack(values, dim=0)


def make_d_dataset_(obs_num, dim, k):
    assert(k <= dim)
    centers = torch.zeros(k, dim)
    for i, c in enumerate(centers):
        c[i] = 10

    dises = []
    for i in range(k):
        loc = centers[i]
        cov = torch.eye(dim) * 2.0
        dis = D.MultivariateNormal(loc=loc, covariance_matrix=cov)
        dises.append(dis)

    values = []
    for _ in range(obs_num // k):
        vs = [dis.sample() for dis in dises]
        values.extend(vs)

    return torch.stack(values, dim=0)


if __name__ == "__main__":
    vs, cs = make_iris_dataset()
    print(vs)
