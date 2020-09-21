#!/usr/bin/env python
# -*- coding:utf-8 -*-
import main
import torch
from torchvision import datasets, transforms
import collections
import pickle


MODEL_PATH = "./mnist_cnn.pt"
OUTPUT_PATH = "./fs.pkl"


def store(fs, ts, rs):
    for (f, t) in zip(fs, ts):
        rs[t.item()].append(f)


def check(results):
    s = 0
    for (k, v) in results.items():
        s += len(v)
    assert(10000 == s)


def extract_features(mdl, dvc, ldr):
    results = collections.defaultdict(list)
    for data, target in ldr:
        data, target = data.to(dvc), target.to(dvc)
        features = mdl.extract(data)
        features = features.cpu().numpy()
        store(features, target, results)
    check(results)
    return results


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda")
    model = main.Net_().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    kwargs = {'batch_size': 64}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, **kwargs)

    model.eval()
    with torch.no_grad():
        results = extract_features(model, device, loader)
        pickle.dump(results, open(OUTPUT_PATH, "bw"))
