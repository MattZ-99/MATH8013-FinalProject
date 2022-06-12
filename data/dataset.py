# -*- coding: utf-8 -*-
# @Time : 2022/5/13 18:35
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# show images
# imshow(torchvision.utils.make_grid(images))
def imshow(img):
    img = img / 2 + 0.5     # normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_class_name_cifar10():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes


def get_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


def get_dataset_cifar10(root='./', transform=None):
    if transform is None:
        # transform_train = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform_test = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_train, transform_test = get_transform()
    else:
        raise NotImplementedError

    train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True, transform=transform_train)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform_test)

    return train_set, test_set


if __name__ == '__main__':
    get_dataset_cifar10()
