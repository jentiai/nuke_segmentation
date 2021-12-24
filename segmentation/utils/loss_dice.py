"""
Reference:
[https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py]
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def coefficient_dice(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += coefficient_dice(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def coefficient_dice_multiclass(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += coefficient_dice(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
    return dice / input.shape[1]


def loss_dice(batch_prediction: Tensor, batch_label: Tensor, count_class: int):
    if count_class > 2:
        multiclass = True
    else:
        multiclass = False

    batch_prediction = F.softmax(batch_prediction, dim=1).float()
    batch_label = F.one_hot(batch_label, count_class).permute(0, 3, 1, 2).float()

    # Dice loss (objective to minimize) between 0 and 1
    assert batch_prediction.size() == batch_label.size()
    fn = coefficient_dice_multiclass if multiclass else coefficient_dice
    return 1 - fn(batch_prediction, batch_label, reduce_batch_first=True)