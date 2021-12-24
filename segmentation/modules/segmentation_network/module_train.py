import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.utils.loss_dice import loss_dice
from segmentation.utils.iou import get_intersection_and_union, print_iou
from segmentation.modules.segmentation_network.module_dataset import JentiDataset
from segmentation.modules.segmentation_network.module_model import get_model


class ModuleTrain:
    def __init__(self, dataio, count_class, size_batch, lr, backbone, num_worker=4, dict_num2class=None):
        self.dataio = dataio
        self.count_class = count_class
        self.size_batch = size_batch
        self.lr = lr
        self.dict_num2class = dict_num2class

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'device' in Pytorch Context = Processing Unit
        self.model = get_model(count_class, backbone=backbone)
        self.model.to(self.device)

        index_min = int(min([dataio.count_total_frames(num_camera) for num_camera in range(dataio.count_device)]))  # 'device' in DataIo Context = Camera
        list_index_train = [i for i in range(0, int(index_min * 0.9))]
        list_index_val = [i for i in range(int(index_min * 0.9), index_min)]

        self.dataset_train = JentiDataset(dataio, backbone=backbone, list_index_frame_to_use=list_index_train)
        self.dataset_val = JentiDataset(dataio, backbone=backbone, list_index_frame_to_use=list_index_val)

        self.loader_train = DataLoader(self.dataset_train, batch_size=size_batch, num_workers=num_worker, pin_memory=True)
        self.loader_val = DataLoader(self.dataset_val, batch_size=size_batch)

        self.optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=lr, weight_decay=1e-8)
        self.loss_cross_entropy = torch.nn.CrossEntropyLoss()
        self.loss_dice = loss_dice

    def train(self, epoch, max_epoch):
        loss_epoch = 0
        arr_intersection_total = np.zeros([self.count_class], dtype=np.float64)
        arr_union_total = np.zeros([self.count_class], dtype=np.float64)

        with tqdm(total=len(self.dataset_train), desc='Epoch {0}/{1}'.format(epoch + 1, max_epoch)) as pbar:
            for batch_color, batch_label in self.loader_train:
                batch_color, batch_label = \
                    batch_color.to(device=self.device, dtype=torch.float32), batch_label.to(device=self.device,
                                                                                            dtype=torch.long)
                batch_prediction = self.model(batch_color)
                loss = self.loss_cross_entropy(batch_prediction, batch_label) + \
                       self.loss_dice(batch_prediction, batch_label, count_class=self.count_class)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                intersection, union = get_intersection_and_union(batch_prediction, batch_label,
                                                                 count_class=self.count_class)
                arr_intersection_total += intersection
                arr_union_total += union

                pbar.update(batch_color.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                loss_epoch += loss.item() / len(self.dataset_train) * self.size_batch

        print('\n[Epoch{0:d}] Loss : {1:5.5f}'.format(epoch + 1, loss_epoch))
        arr_intersection_total[arr_union_total == 0] = -1
        arr_union_total[arr_union_total == 0] = 1
        arr_iou = arr_intersection_total / arr_union_total
        print('[Training IoU]')
        print_iou(arr_iou, self.dict_num2class)

    def validate(self):
        arr_intersection_total = np.zeros([self.count_class], dtype=np.float64)
        arr_union_total = np.zeros([self.count_class], dtype=np.float64)

        for batch_color, batch_label in self.loader_val:
            batch_color, batch_label = \
                batch_color.to(device=self.device, dtype=torch.float32), batch_label.to(device=self.device,
                                                                                        dtype=torch.long)
            batch_prediction = self.model(batch_color)

            intersection, union = get_intersection_and_union(batch_prediction, batch_label,
                                                             count_class=self.count_class)
            arr_intersection_total += intersection
            arr_union_total += union

        arr_intersection_total[arr_union_total == 0] = -1
        arr_union_total[arr_union_total == 0] = 1
        arr_iou = arr_intersection_total / arr_union_total
        print('[Validation IoU]')
        print_iou(arr_iou, self.dict_num2class)

    def run(self, max_epoch, freq_save=1, freq_val=1, dir_checkpoint='ckpt_Enetb4_FPN'):
        self.model.train()
        for epoch in range(max_epoch):
            self.train(epoch, max_epoch)

            if (epoch + 1) % freq_val == 0:
                self.validate()

            if (epoch + 1) % freq_save == 0:
                torch.save(self.model.state_dict(), os.path.join(dir_checkpoint, 'epoch%d.pth' % (epoch + 1)))
