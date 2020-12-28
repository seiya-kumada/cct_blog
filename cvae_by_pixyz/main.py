#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torchvision import transforms
from torchvision import datasets
import torch
import net
from pixyz.distributions import Normal
from pixyz.losses import KullbackLeibler
from pixyz.losses import Expectation
from pixyz.losses import LogProb
from pixyz.models import Model
from torch import optim
# from torchvision.utils import save_image
# import os
from tensorboardX import SummaryWriter
import datetime
import pixyz


ROOT_DIR_PATH = './data'
BATCH_SIZE = 128
EPOCHS = 1
SEED = 1
CLASS_SIZE = 10
SAVE_ROOT_DIR_PATH = './results'


def learn(epoch, model, device, loader, phase):
    sum_loss = 0
    for batch_idx, (x, y) in enumerate(loader, 1):
        x = x.to(device)
        y = torch.eye(CLASS_SIZE)[y].to(device)
        loss = model.train({"x": x, "y": y})
        sum_loss += loss

    sum_loss = sum_loss * loader.batch_size / len(loader.dataset)
    print('Epoch: {} {} loss: {:.4f}'.format(epoch, phase, sum_loss))
    return sum_loss


def plot_reconstruction(x, y, p, q):
    with torch.no_grad():
        # q(z|x,y)
        z = q.sample({"x": x, "y": y}, return_all=False)
        z.update({"y": y})

        # p(x|z,y)
        recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)

        recon = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
        return recon


def generate(z, y, p):
    with torch.no_grad():
        # p(x|y,z)
        sample = p.sample_mean({"z": z, "y": y}).view(-1, 1, 28, 28).cpu()
        return sample


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 1次元ベクトルに変更
        transforms.Lambda(lambd=lambda x: x.view(-1))])

    # pin_memoryをTrueにすると(Automatic Memory Pinning)
    # CPUのメモリ領域がページングされなくなり高速化を期待できる。
    kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=ROOT_DIR_PATH, train=True, transform=transform, download=True),
        shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=ROOT_DIR_PATH, train=False, transform=transform),
        shuffle=False, **kwargs)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # p(x|y,z)
    p = net.Generator().to(device)

    # q(z|x,y)
    q = net.Inference().to(device)

    # prior p(z)
    prior = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0),
                   var=["z"], features_shape=[net.Z_DIM], name="p_{prior}").to(device)

    # print(p)
    # print(q)
    # print(prior)

    loss = (KullbackLeibler(q, prior) - Expectation(q, LogProb(p))).mean()
    model = Model(loss=loss, distributions=[p, q], optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    # print(model)

    _x, _y = next(iter(test_loader))
    _x = _x.to(device)
    _y = torch.eye(CLASS_SIZE)[_y].to(device)

    dt_now = datetime.datetime.now()
    exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
    v = pixyz.__version__
    nb_name = 'cvae'
    writer = SummaryWriter("runs/" + v + "." + nb_name + exp_time)

    for epoch in range(1, EPOCHS + 1):
        train_loss = learn(epoch, model, device, train_loader, "Train")
        test_loss = learn(epoch, model, device, test_loader, "Test")

        recon = plot_reconstruction(_x[:8], _y[:8], p, q)
        writer.add_images('Image_reconstrunction', recon, epoch)

        # save_image(
        #     torch.cat([_x[:8], recon], dim=0),
        #     os.path.join(SAVE_ROOT_DIR_PATH, "reconst_{}.png".format(epoch)),
        #     nrow=8)

        # gen = generate(z_sample, y_sample, p)
        # save_image(
        #     gen,
        #     os.path.join(SAVE_ROOT_DIR_PATH, "gen_{}.png".format(epoch)),
        #     nrow=8)
