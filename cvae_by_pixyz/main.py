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
from torchvision.utils import save_image
import os
from matplotlib import pyplot as plt

ROOT_DIR_PATH = './data'
BATCH_SIZE = 128
EPOCHS = 10
SEED = 1
CLASS_SIZE = 10
SAVE_ROOT_DIR_PATH = './results'

torch.manual_seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def learn(eph, mdl, dvc, ldr, phs):
    # モードの切り替え
    if phs is "Train":
        learning_process = mdl.train
    else:
        learning_process = mdl.test

    log_interval = len(ldr) // 10   # 進捗を表示する間隔
    if log_interval == 0:
        log_interval = 1

    loss = 0
    for batch_idx, (x, y) in enumerate(ldr, 1):
        x = x.to(dvc)
        y = torch.eye(CLASS_SIZE)[y].to(dvc)
        running_loss = learning_process({"x": x, "y": y})
        loss += running_loss

        if batch_idx % log_interval == 0:
            print(f'{"train" if phs == "Train" else "test"}... [Epoch:{epoch}/Batches:{batch_idx}] loss: {running_loss / len(x):.4f}')

    loss = loss * ldr.batch_size / len(ldr.dataset)
    # print('Epoch: {} {} loss: {:.4f}'.format(eph, phs, sum_loss))
    return loss.item()


def reconstruct_image(p, q, x, y):
    with torch.no_grad():
        # q(z|x,y)
        z = q.sample({"x": x, "y": y}, return_all=False)
        z.update({"y": y})

        # p(x|z,y)
        x_reconst = p.sample_mean(z)

        return x_reconst.view(-1, 1, 28, 28)


def generate(z, y, p):
    with torch.no_grad():
        # p(x|y,z)
        sample = p.sample_mean({"z": z, "y": y}).view(-1, 1, 28, 28).cpu()
        return sample


def plot_figure(eph, train_losses, test_losses):
    # 損失値のグラフを作成し保存
    plt.plot(list(range(1, eph + 1)), train_losses, label='train')
    plt.plot(list(range(1, eph + 1)), test_losses, label='test')
    plt.xlabel('ephs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_ROOT_DIR_PATH, 'loss.png'))
    plt.close()


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

    loss = (KullbackLeibler(q, prior) - Expectation(q, LogProb(p))).mean()
    model = Model(loss=loss, distributions=[p, q], optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    # print(model)

    x_fixed, y_fixed = next(iter(test_loader))
    x_fixed = x_fixed[:8].to(device)
    y_fixed = y_fixed[:8]
    y_fixed = torch.eye(CLASS_SIZE)[y_fixed].to(device)
    z_fixed = prior.sample(batch_n=64)['z']

    train_loss_list = []
    test_loss_list = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = learn(epoch, model, device, train_loader, "Train")
        test_loss = learn(epoch, model, device, test_loader, "Test")
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print(f'    [Epoch {epoch}] train loss {train_loss_list[-1]:.4f}')
        print(f'    [Epoch {epoch}] test  loss {test_loss_list[-1]:.4f}\n')

        x_reconst = reconstruct_image(p, q, x_fixed, y_fixed)
        save_image(
            torch.cat([x_fixed.view(-1, 1, 28, 28), x_reconst], dim=0),
            os.path.join(SAVE_ROOT_DIR_PATH, f'reconst_{epoch}.png'), nrow=8)

        plot_figure(epoch, train_loss_list, test_loss_list)
        # gen = generate(z_sample, y_sample, p)
        # save_image(
        #     gen,
        #     os.path.join(SAVE_ROOT_DIR_PATH, "gen_{}.png".format(epoch)),
        #     nrow=8)
