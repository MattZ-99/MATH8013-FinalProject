# -*- coding: utf-8 -*-
# @Time : 2022/5/17 20:16
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""
import numpy as np
import torch


def get_accuracy(pred, actual):
    """Calculate the accuracy."""

    assert len(pred) == len(actual)

    total = len(actual)
    _, predicted = torch.max(pred.data, 1)
    correct = (predicted == actual).float().sum().item()
    return correct / total


def calculate_confusion_matrix(pred, actual, matrix_size):
    """Calculate the confusion matrix, with the predicted and actual labels."""

    matrix = np.zeros(shape=(matrix_size, matrix_size))
    np.add.at(matrix, (actual, pred), 1)

    matrix_sum = np.sum(matrix, axis=1)

    # matrix normalize
    return matrix / matrix_sum[:, None]
