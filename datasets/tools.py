import numpy as np
import random
import torch
from PIL import Image


from math import inf
import torch.nn.functional as F
import torch.nn as nn
def load_image(idx):
    data_file = open('./data/tiny_images.bin', "rb")

    data_file.seek(idx * 3072)
    data = data_file.read(3072)
    return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")


def sample_id(data_num=50000, exclude_cifar=True):
    id_list = list(range(79302017))

    if exclude_cifar:
        cifar_idxs = []
        with open('./datasets/80mn_cifar_idxs.txt', 'r') as idxs:
            for idx in idxs:
                cifar_idxs.append(int(idx) - 1)
        cifar_idxs = set(cifar_idxs)
        id_no_cifar = [x for x in id_list if x not in cifar_idxs]

        id_sample = random.sample(id_no_cifar, data_num)
    else:
        id_sample = random.sample(id_list, data_num)

    return id_sample


def load_open_data(dataset):
    if dataset == "cifar10g":
        process_func = lambda x: ((x * 0.5 + 0.5) * 255).astype(np.uint8)
        data = process_func(np.load("./data/unconditional_cifar10_samples.npy"))
    elif dataset == "cifar5m":
        data = np.load("./data/cifar5m/cifar5m_sampled.npy", mmap_mode='r')

    return data


class DependentLabelGenerator:
    def __init__(self, num_classes, feature_size, seed, transform):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.W = torch.FloatTensor(np.random.randn(num_classes, feature_size, num_classes))
        self.num_classes = num_classes
        self.seed = seed
        self.transform = transform
    def generate_dependent_labels(self, data, target):
        # 1*m *  m*10 = 1*10
        img = Image.fromarray(data)
        img = self.transform(img)
        A = img.view(1, -1).mm(self.W[target]).squeeze(0)
        A[target] = -inf
        A = F.softmax(A, dim=0)

        new_label = int(np.random.choice(list(range(self.num_classes)), p=A.cpu().numpy()))
        return new_label