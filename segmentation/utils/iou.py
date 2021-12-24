import torch
from torch import Tensor
import numpy as np
from numba import jit


@jit(nopython=True)
def calculate_intersection_and_union(batch_prediction: np.ndarray, batch_label: np.ndarray, count_class: int):
    arr_intersection = np.zeros(shape=(count_class,), dtype=np.float64)
    arr_union = np.zeros(shape=(count_class,), dtype=np.float64)
    for num_class in range(count_class):
        bool_prediction = batch_prediction == num_class
        bool_label = batch_label == num_class

        intersection = np.sum((bool_prediction & bool_label).astype(np.float64))
        union = np.sum((bool_prediction | bool_label).astype(np.float64))

        arr_intersection[num_class] = intersection
        arr_union[num_class] = union
    return arr_intersection, arr_union


def get_intersection_and_union(batch_prediction: Tensor, batch_label: Tensor, count_class: int):
    batch_prediction = torch.argmax(batch_prediction, dim=1).detach().cpu().numpy()
    batch_label = batch_label.detach().cpu().numpy()
    arr_intersection, arr_union = calculate_intersection_and_union(batch_prediction, batch_label, count_class)
    return arr_intersection, arr_union


def print_iou(arr_iou: np.ndarray, dict_num2class):
    for num_class, iou in enumerate(arr_iou):
        if iou == -1:
            continue
        if dict_num2class is not None:
            print('IoU - {0:15s} : {1:1.3f}'.format(dict_num2class[num_class], iou))
        else:
            print('IoU - {0:15d} : {1:1.3f}'.format(num_class, iou))