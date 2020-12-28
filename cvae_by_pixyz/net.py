#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from pixyz.distributions import Normal
from pixyz.distributions import Bernoulli
from torch import nn
from torch.nn import functional as F


X_DIM = 784
Y_DIM = 10
Z_DIM = 64
H_DIM = 512


# q(z|x,y)
class Inference(Normal):

    def __init__(self):
        super().__init__(var=["z"], cond_var=["x", "y"], name="q")

        self.fc1 = nn.Linear(X_DIM + Y_DIM, H_DIM)
        self.fc2 = nn.Linear(H_DIM, H_DIM)
        self.fc31 = nn.Linear(H_DIM, Z_DIM)
        self.fc32 = nn.Linear(H_DIM, Z_DIM)

    def forward(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], 1)))
        h = F.relu(self.fc2(h))
        # scale is variance
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


# p(z|z,y)
class Generator(Bernoulli):

    def __init__(self):
        super().__init__(var=["x"], cond_var=["z", "y"], name="p")

        self.fc1 = nn.Linear(Z_DIM + Y_DIM, H_DIM)
        self.fc2 = nn.Linear(H_DIM, H_DIM)
        self.fc3 = nn.Linear(H_DIM, X_DIM)

    def forward(self, z, y):
        h = F.relu(self.fc1(torch.cat([z, y], 1)))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}
