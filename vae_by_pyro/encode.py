#!/usr/bin/env python
# -*- coding:utf-8 -*-

import vae_model
import argparse
from utils.vae_plots import plot_llk  # , plot_vae_samples
from pyro.optim import Adam
import pyro
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
import numpy as np
import myutils
import custom_dataset as cd
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

matplotlib.use('Agg')
INPUT_DIR_PATH = "/home/ubuntu/data/mitsubishi_motors/isu_detection/pattern_2/train/patches_25_with_blob_positions"
TEST_DIR_PATH = "/home/ubuntu/data/mitsubishi_motors/isu_detection/pattern_4/ng_patches"
MODEL_PATH = "./vae.pth"
IMAGE_SIZE = 25
DATA_SIZE = 25 * 25  # 784
BATCH_SIZE = 200


def plot_tsne(train_z_locs, test_z_locs):
    model_tsne = TSNE(n_components=2, random_state=0)

    train_z_states = train_z_locs.detach().cpu().numpy()
    test_z_states = test_z_locs.detach().cpu().numpy()
    all_z_states = np.concatenate([train_z_states, test_z_states], axis=0)
    train_size, _ = train_z_states.shape
    test_size, _ = test_z_states.shape
    colors = [0] * train_size + [1] * test_size

    all_z_embed = model_tsne.fit_transform(all_z_states)

    fig = plt.figure()
    plt.scatter(all_z_embed[:, 0], all_z_embed[:, 1], s=10, c=colors, alpha=0.1)
    plt.title("0: Train, 1: Test")
    plt.colorbar()
    fig.savefig('./vae_results/embedding.png')


def draw_distributions(vae, train_loader, test_loader, is_cuda):
    train_z_locs = extract_z_locs(vae, train_loader, is_cuda, 10)
    test_z_locs = extract_z_locs(vae, test_loader, is_cuda, 10)
    plot_tsne(train_z_locs, test_z_locs)


def extract_z_locs(vae, loader, is_cuda, num, is_all=False):
    z_locs = []
    for i, (x, _) in enumerate(loader):
        if is_cuda:
            x = x.cuda()
        z_loc, z_scale = vae.encoder(x)
        z_locs.append(z_loc)
        if (not is_all) and (i == num):
            break
    z_locs = torch.cat(z_locs, dim=0)
    return z_locs


def draw_distribution(vae, loader, is_cuda):
    z_locs = []
    for i, (x, _) in enumerate(loader):
        if is_cuda:
            x = x.cuda()
        z_loc, z_scale = vae.encoder(x)
        z_locs.append(z_loc)
        if i == 10:
            break
    z_locs = torch.cat(z_locs, dim=0)
    plot_tsne(z_locs)


def main(args):
    # clear param store
    pyro.clear_param_store()

    train_paths = myutils.load_images(INPUT_DIR_PATH)
    train_dataset = cd.CustomDataset(IMAGE_SIZE, train_paths)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("> custom_datasize:{}".format(len(train_dataset)))
    print("> custom_loader:{}".format(len(train_loader)))

    # setup the VAE
    vae = vae_model.VAE(data_size=DATA_SIZE, z_dim=20, use_cuda=args.cuda)

    # load the trained model
    vae.load_state_dict(torch.load(MODEL_PATH))

    test_paths = myutils.load_images(TEST_DIR_PATH)
    test_dataset = cd.CustomDataset(IMAGE_SIZE, test_paths)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    draw_distributions(vae=vae, train_loader=train_loader, test_loader=test_loader, is_cuda=args.cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    # parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    # parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    # parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    # parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    # parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    args = parser.parse_args()

    model = main(args)
