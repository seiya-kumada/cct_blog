#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer import TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro import poutine
from pyro.infer import SVI
import sys
# from torch.distributions import constraints

DIR_PATH = "/home/ubuntu/data/cct_blog/pyro/train/"
LABEL_MAP = {"pin": 0, "block": 1, "sheet_metal": 2}


# consider three gaussians
K = 3


def load_data(dir_path):
    data = []
    for line in os.listdir(dir_path):
        if "._" in line:
            continue
        path = os.path.join(dir_path, line)
        data.append(np.load(path))
    return torch.tensor(data)


# @pyro.infer.config_enumerate(default='parallel')
# @pyro.poutine.broadcast
# def full_guide(data):
#     with poutine.block(hide_types=["param"]):
#         global_guide(data)
#
#     with pyro.plate('data', len(data)):
#         assignment_probs = pyro.param(
#             'assignment_probs',
#             torch.ones(len(data)) / K,
#             constraint=constraints.unit_interval)
#         pyro.sample('assignments', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})


@pyro.infer.config_enumerate(default='parallel')
@pyro.poutine.broadcast
def model(data):
    _, dim = data.shape
    weights = pyro.sample('weights', dist.Dirichlet(torch.ones(K)))

    with pyro.plate('c1', K):
        mus = pyro.sample('mus', dist.MultivariateNormal(torch.zeros(dim), torch.diag(torch.ones(dim) * 10.0)))
    assert(mus.size() == (K, dim))

    with pyro.plate('dim', dim):
        with pyro.plate('c2', K):
            lambdas = pyro.sample('lambdas', dist.LogNormal(0, 2))
    assert(lambdas.size() == (K, dim))

    scales = []
    for k in range(K):
        scales.append(torch.diag(lambdas[k]))
    scales = torch.stack(scales, dim=0)
    assert((K, dim, dim) == scales.size())

    with pyro.plate('data', len(data)):
        assignments = pyro.sample('assignments', dist.Categorical(weights))
        pyro.sample(
            'obs',
            dist.MultivariateNormal(
                mus[assignments],
                scales[assignments]
            ),
            obs=data
        )


def run_mcmc(data):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=250, warmup_steps=50)
    mcmc.run(data)


def initialize(data):
    pyro.clear_param_store()
    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=2)
    # global global_guide
    global_guide = AutoDelta(poutine.block(model, expose=['weights', 'mus', 'lambdas']))
    svi = SVI(model, global_guide, optim, loss=elbo)
    svi.loss(model, global_guide, data)
    return svi


def run_map(data):
    svi = initialize(data)
    losses = []
    for i in range(200):
        loss = svi.step(data)
        losses.append(loss)
        print('.' if i % 100 else '\n', end='')
        sys.stdout.flush()
    print("")


if __name__ == "__main__":
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    N = 912
    M = 10
    data = load_data(DIR_PATH)[:N, :M]
    assert(data.size() == (N, M))

    # run_map(data)
    run_mcmc(data)
