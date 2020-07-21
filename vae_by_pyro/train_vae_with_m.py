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

INPUT_DIR_PATH = "/home/ubuntu/data/mitsubishi_motors/isu_detection/pattern_2/train/patches_25_with_blob_positions"
IMAGE_SIZE = 25
DATA_SIZE = 25 * 25  # 784
BATCH_SIZE = 200


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

        if epoch == args.tsne_iter:
            print("HHH", epoch)
            # mnist_test_tsne(vae=vae, test_loader=test_loader)
            plot_llk(np.array(train_elbo), np.array(test_elbo))

    return vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)
