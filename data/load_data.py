# -*- coding: utf-8 -*-
# @Time : 2022/5/13 17:27
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        res_dict = pickle.load(fo, encoding='bytes')
    return res_dict


def load_data_cifar10(root_dir='./cifar-10-batches-py'):
    train_data_dict_list = list()
    test_data_dict_list = list()

    for file_name in sorted(os.listdir(root_dir)):
        if file_name in ['batches.meta', 'readme.html']:
            continue

        data_dict = unpickle(os.path.join(root_dir, file_name))
        if data_dict[b'batch_label'] == b'testing batch 1 of 1':
            test_data_dict_list.append(data_dict)
        else:
            train_data_dict_list.append(data_dict)
    return train_data_dict_list, test_data_dict_list


if __name__ == '__main__':
    load_data_cifar10()
