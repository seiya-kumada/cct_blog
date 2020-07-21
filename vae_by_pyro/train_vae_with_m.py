#!/usr/bin/env python
# -*- coding:utf-8 -*-

import vae_model
import argparse
# from utils.vae_plots import mnist_test_tsne, plot_llk  # , plot_vae_samples
from utils.vae_plots import plot_llk  # , plot_vae_samples
from pyro.optim import Adam
import pyro
# from utils.mnist_cached import setup_data_loaders
# from utils.mnist_cached import MNISTCached as MNIST
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
# import visdom
import numpy as np
import myutils
import custom_dataset as cd
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

matplotlib.use('Agg')
INPUT_DIR_PATH = "/home/ubuntu/data/mitsubishi_motors/isu_detection/pattern_2/train/patches_25_with_blob_positions"
TEST_DIR_PATH = "/home/ubuntu/data/mitsubishi_motors/isu_detection/test/ng/patches_25"
IMAGE_SIZE = 25
DATA_SIZE = 25 * 25  # 784
BATCH_SIZE = 200


def plot_tsne(train_z_locs, test_z_locs):
    model_tsne = TSNE(n_components=2, random_state=0)

    train_z_states = train_z_locs.detach().cpu().numpy()
    train_z_embed = model_tsne.fit_transform(train_z_states)

    test_z_states = test_z_locs.detach().cpu().numpy()
    test_z_embed = model_tsne.fit_transform(test_z_states)

    fig = plt.figure()
    plt.scatter(train_z_embed[:, 0], train_z_embed[:, 1], s=10, label="train")
    plt.scatter(test_z_embed[:, 0], test_z_embed[:, 1], s=10, label="test")
    plt.title("Latent Variable T-SNE per Class")
    plt.legend(loc="best")
    fig.savefig('./vae_results/embedding.png')


def draw_distributions(vae, train_loader, test_loader, is_cuda):
    train_z_locs = extract_z_locs(vae, train_loader, is_cuda, 10)
    test_z_locs = extract_z_locs(vae, test_loader, is_cuda, 10)
    plot_tsne(train_z_locs, test_z_locs)


def extract_z_locs(vae, loader, is_cuda, num):
    z_locs = []
    for i, (x, _) in enumerate(loader):
        if is_cuda:
            x = x.cuda()
        z_loc, z_scale = vae.encoder(x)
        z_locs.append(z_loc)
        if i == num:
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

    # setup MNIST data loaders
    # train_loader, test_loader
    # train_loader, test_loader = setup_data_loaders(MNIST, use_cuda=args.cuda, batch_size=256)

    paths = myutils.load_images(INPUT_DIR_PATH)
    train_dataset = cd.CustomDataset(IMAGE_SIZE, paths)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("> custom_datasize:{}".format(len(train_dataset)))
    print("> custom_loader:{}".format(len(train_loader)))

    # setup the VAE
    vae = vae_model.VAE(data_size=DATA_SIZE, use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    # setup visdom for visualization
    # if args.visdom_flag:
    #     vis = visdom.Visdom()

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(train_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

        #         # pick three random test images from the first mini-batch and
        #         # visualize how well we're reconstructing them
        #         if i == 0:
        #             if args.visdom_flag:
        #                 plot_vae_samples(vae, vis)
        #                 reco_indices = np.random.randint(0, x.shape[0], 3)
        #                 for index in reco_indices:
        #                     test_img = x[index, :]
        #                     reco_img = vae.reconstruct_img(test_img)
        #                     vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
        #                               opts={'caption': 'test image'})
        #                     vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
        #                               opts={'caption': 'reconstructed image'})

            # report test diagnostics
            normalizer_test = len(train_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

    # draw_distribution(vae=vae, loader=train_loader, is_cuda=args.cuda)
    plot_llk(np.array(train_elbo), np.array(test_elbo))
    torch.save(vae.state_dict(), "./vae.pth")

    paths = myutils.load_images(TEST_DIR_PATH)
    test_dataset = cd.CustomDataset(IMAGE_SIZE, paths)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    draw_distributions(vae=vae, train_loader=train_loader, test_loader=test_loader, is_cuda=args.cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    args = parser.parse_args()

    model = main(args)
