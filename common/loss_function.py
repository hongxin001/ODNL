import torch.nn as nn
import torch

def gce_loss(outputs, labels):
    q = 0.7
    k = 10
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    label_one_hot = nn.functional.one_hot(labels, k).float().cuda()
    sm_out = torch.pow((sm_outputs * label_one_hot).sum(dim=1), q)
    target = torch.ones_like(labels)
    loss_vec = (target - sm_out)/q
    average_loss = loss_vec.mean()
    return average_loss