import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    n=len(dataset)
    for i, b in enumerate(dataset):
        # img = b[0]
        # true_mask = b[1]
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #
        # if gpu:
        #     img = img.to(device)
        #     true_mask = true_mask.to(device)

        img = torch.from_numpy(b[0]).unsqueeze(0).float()
        label = torch.from_numpy(b[1]).unsqueeze(0).long()
        if gpu:
            img = img.to(device)
            label = label.to(device)

        pred = net(img)
        loss = nn.CrossEntropyLoss()
        loss = loss(pred, label)

        tot += loss.item()
    return tot / n

def eval_net_BCE(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    n=len(dataset)
    for i, b in enumerate(dataset):
        # img = b[0]
        # true_mask = b[1]
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #
        # if gpu:
        #     img = img.to(device)
        #     true_mask = true_mask.to(device)

        img = torch.from_numpy(b[0]).unsqueeze(0).float()
        label = torch.from_numpy(b[1]).unsqueeze(0).float()
        if gpu:
            img = img.to(device)
            label = label.to(device)

        pred = net(img)
        pred_flat = pred.view(-1)
        labels_flat = label.view(-1)
        loss = nn.BCEWithLogitsLoss()
        loss = loss(pred_flat, labels_flat)

        tot += loss.item()
    return tot / n
