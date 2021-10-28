from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys, json

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import random
import torch.utils.data as data
import torch
from datasets.tools import DependentLabelGenerator

from common.utils import download_url, check_integrity

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, dataset_type="train",
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type  # training set or test set
        self.dataset = 'cifar10'
        self.noise_type = noise_type
        self.nb_classes = 10
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if dataset_type != "test":
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            # self.train_labels = torch.Tensor(self.train_labels).long()

            # (self.train_data, self.train_labels), (self.valid_data, self.valid_labels) = train_valid_split((torch.from_numpy(self.train_data),self.train_labels))
            self.train_labels = np.asarray(self.train_labels)

            # if noise_type is not None:
            if dataset_type == "train":
                # self.train_data = self.train_data.numpy()
                if noise_type != 'clean':
                    # noisify train data
                    self.train_noisy_labels = self.train_labels[:]
                    noise_file = '%s/%s_%s_%.1f.json' % ("./datasets/config", self.dataset, noise_type, noise_rate)
                    if os.path.exists(noise_file):
                        noise_data = json.load(open(noise_file, "r"))

                        if self.noise_type == 'open':
                            noise_idx, open_idx = zip(*noise_data)
                            noise_idx, open_idx = list(noise_idx), list(open_idx)

                            open_dataset = CIFAR100(root='./data/',
                                                    download=True,
                                                    dataset_type="train",
                                                    noise_type='clean',
                                                    noise_rate=0
                                                    )
                            self.train_data[noise_idx] = open_dataset.train_data[open_idx]
                        else:
                            self.train_noisy_labels = np.array(noise_data)
                    else:  # inject noise
                        # noise_label = []
                        idx = list(range(50000))
                        random.shuffle(idx)
                        num_noise = int(noise_rate * 50000)
                        noise_idx = idx[:num_noise]
                        label_gen = DependentLabelGenerator(self.nb_classes, 32 * 32 * 3, random_state, self.transform)

                        if self.noise_type == 'open':
                            open_dataset = CIFAR100(root='./data/',
                                                    download=True,
                                                    dataset_type="train",
                                                    noise_type='clean',
                                                    noise_rate=0
                                                    )
                            open_idx = np.random.choice(len(open_dataset), len(noise_idx),
                                                        replace=False).tolist()
                            self.train_data[noise_idx] = open_dataset.train_data[open_idx]
                            self.open_noise = list(zip(noise_idx, open_idx))
                            print("save noisy data to %s ..." % noise_file)
                            json.dump(self.open_noise, open(noise_file, "w"))
                        else:
                            for i in noise_idx:
                                if self.noise_type == 'symmetric':
                                    noisy_prob = (1 - np.eye(self.nb_classes)[self.train_labels[i]]) / (
                                                self.nb_classes - 1)
                                    noiselabel = int(np.random.choice(list(range(self.nb_classes)), p=noisy_prob))
                                elif self.noise_type == 'asymmetric':
                                    noiselabel = self.transition[self.train_labels[i]]
                                elif self.noise_type == 'dependent':
                                    noiselabel = label_gen.generate_dependent_labels(self.train_data[i],
                                                                                     self.train_labels[i])
                                self.train_noisy_labels[i] = noiselabel
                            print("save noisy labels to %s ..." % noise_file)
                            json.dump(self.train_noisy_labels.tolist(), open(noise_file, "w"))
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)

                else:
                    self.noise_or_not = None
            # else:
            #     self.valid_data = self.valid_data.numpy()


        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.dataset_type == "train":
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        # elif self.dataset_type == "valid":
        #     img, target = self.valid_data[index], self.valid_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.dataset_type == "train":
            return len(self.train_data)

        # elif self.dataset_type == "valid":
        #     return len(self.valid_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, dataset_type="train",
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type  # training set or test set
        self.dataset = 'cifar100'
        self.noise_type = noise_type
        self.nb_classes = 100
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if dataset_type != "test":
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            # self.train_labels = torch.Tensor(self.train_labels).long()

            # (self.train_data, self.train_labels), (self.valid_data, self.valid_labels) = train_valid_split((torch.from_numpy(self.train_data),self.train_labels))
            self.train_labels = np.asarray(self.train_labels)

            # if noise_type is not None:
            if dataset_type == "train":
                # self.train_data = self.train_data.numpy()
                if noise_type != 'clean':
                    # noisify train data
                    self.train_noisy_labels = self.train_labels[:]
                    noise_file = '%s/%s_%s_%.1f.json' % ("./datasets/config", self.dataset, noise_type, noise_rate)
                    if os.path.exists(noise_file):
                        noise_data = json.load(open(noise_file, "r"))

                        if self.noise_type == 'open':
                            noise_idx, open_idx = zip(*noise_data)
                            noise_idx, open_idx = list(noise_idx), list(open_idx)
                            open_dataset = CIFAR10(root='./data/',
                                                   download=True,
                                                   dataset_type="train",
                                                   noise_type='clean',
                                                   noise_rate=0
                                                   )
                            self.train_data[noise_idx] = open_dataset.train_data[open_idx]
                        else:
                            self.train_noisy_labels = np.array(noise_data)
                    else:
                        # inject noise
                        # noise_label = []
                        idx = list(range(50000))
                        random.shuffle(idx)
                        num_noise = int(noise_rate * 50000)
                        noise_idx = idx[:num_noise]
                        label_gen = DependentLabelGenerator(self.nb_classes, 32 * 32 * 3, random_state, self.transform)

                        if self.noise_type == 'open':
                            open_dataset = CIFAR10(root='./data/',
                                                   download=True,
                                                   dataset_type="train",
                                                   noise_type='clean',
                                                   noise_rate=0
                                                   )
                            open_idx = np.random.choice(len(open_dataset), len(noise_idx),
                                                        replace=False).tolist()
                            self.train_data[noise_idx] = open_dataset.train_data[open_idx]
                            self.open_noise = list(zip(noise_idx, open_idx))
                            print("save noisy data to %s ..." % noise_file)
                            json.dump(self.open_noise, open(noise_file, "w"))
                        else:
                            for i in noise_idx:
                                if self.noise_type == 'symmetric':
                                    noisy_prob = (1 - np.eye(self.nb_classes)[self.train_labels[i]]) / (
                                                self.nb_classes - 1)
                                    noiselabel = int(np.random.choice(list(range(self.nb_classes)), p=noisy_prob))
                                elif self.noise_type == 'asymmetric':
                                    noiselabel = (self.train_labels[i] + 1) % self.nb_classes
                                elif self.noise_type == 'dependent':
                                    noiselabel = label_gen.generate_dependent_labels(self.train_data[i],
                                                                                     self.train_labels[i])
                                self.train_noisy_labels[i] = noiselabel
                            print("save noisy labels to %s ..." % noise_file)
                            json.dump(self.train_noisy_labels.tolist(), open(noise_file, "w"))

                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)

                else:
                    self.noise_or_not = None
            # else:
            #     self.valid_data = self.valid_data.numpy()


        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

