from algorithms.base_framework import SingleModel
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from datasets.utils import build_dataset, build_ood_noise

class ODNL(SingleModel):

    def __init__(self, args, device, num_classes, train_loader):
        super(ODNL, self).__init__(args, device, num_classes, train_loader)
        if args.aux_set not in ["Gaussian", "Rademacher", "Blob"]:
            self.ood_data, _ = build_dataset(args, args.aux_set, "train", data_num=args.aux_size, origin_dataset=args.dataset)
        else:
            self.ood_data = build_ood_noise(args.aux_set, args.aux_size, 1)

        self.train_loader_out = torch.utils.data.DataLoader(
            self.ood_data,
            batch_size=args.aux_batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)


    def train(self, train_loader, epoch):
        self.net.train()
        loss_avg = 0.0
        for in_set, out_set in zip(train_loader, self.train_loader_out):
            loss = self.train_batch_with_out(in_set, out_set, epoch)

            # backward
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()
            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        self.scheduler.step()

        return loss_avg


    def train_batch_with_out(self, in_set, out_set, epoch):
        in_data, out_data, target = in_set[0].to(self.device), out_set[0].to(self.device), in_set[1].to(self.device)
        inputs = torch.cat([in_data, out_data], dim=0)
        target_random = torch.LongTensor(out_data.shape[0]).random_(0, self.num_classes).to(self.device)
        logits = self.net(inputs)
        loss = self.loss_function(logits[:in_data.shape[0]], target) + self.args.lambda_o * F.cross_entropy(logits[in_data.shape[0]:], target_random)
        return loss
