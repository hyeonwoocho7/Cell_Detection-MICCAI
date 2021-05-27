import torch.nn as nn
import torch
import cv2
import numpy as np


def eval_net(net, dataset, device, gpu=True, loss=nn.BCEWithLogitsLoss()):
    criterion = loss
    net.eval()
    losses = 0
    acces = 0
    torch.cuda.empty_cache()
    for iteration, data in enumerate(dataset):
        img = data["image"]
        mask = data["gt"]
        label = data["label"]

        if gpu:
            img = img.to(device)
            mask = mask.to(device)
            label = label.float().to(device)

        input = torch.cat([img, mask], dim=1)
        output = net(input)

        loss = criterion(output, label)
        acc = ((torch.sigmoid(output) > 0.5) == label.byte()).float().mean()
        losses += loss.item()
        acces += acc.item()
    return losses / (iteration+1), acces / (iteration+1)