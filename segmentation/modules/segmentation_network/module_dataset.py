import cv2
import numpy as np
import torch
import torchvision.transforms as t
from torch.utils.data import Dataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class JentiDataset(Dataset):
    def __init__(self, dataio, backbone, list_index_frame_to_use=None):
        self.dataio = dataio
        self.h, self.w = dataio.load_semantic_label(0, 0).shape
        self.w_for_inference = self.w - self.w % 32
        self.h_for_inference = self.h - self.h % 32
        self.count_device = dataio.count_device

        self.preprocessing_fn = get_preprocessing_fn(backbone, pretrained="imagenet")

        if list_index_frame_to_use is None:
            index_min = int(min([dataio.count_total_frames(num_device) for num_device in range(self.count_device)]))
            self.list_index_rearranged = [index for index in range(index_min * self.count_device)]
        else:
            self.list_index_rearranged = []
            for index_frame_to_use in list_index_frame_to_use:
                for num_device in range(self.count_device):
                    self.list_index_rearranged.append(index_frame_to_use * self.count_device + num_device)

    def __len__(self):
        return len(self.list_index_rearranged)  # count_frame * count_device

    def __getitem__(self, index):
        num_frame = self.list_index_rearranged[index] // self.count_device
        num_device = self.list_index_rearranged[index] % self.count_device
        image_color = self.dataio.load_color(num_device, num_frame)  # BGR Image : [C, H, W]
        image_semantic_label = self.dataio.load_semantic_label(num_device, num_frame, label_unknown_as_other=True)  # Semantic Label : [H, W]
        return self.preprocessing_fn_color(image_color), self.preprocessing_fn_semantic(image_semantic_label)

    def preprocessing_fn_color(self, image_color):
        image_color = self.preprocessing_fn(image_color)
        image_color = torch.from_numpy(np.transpose(image_color, [2, 0, 1])).float()

        resize_fn = t.Resize([self.h_for_inference, self.w_for_inference], interpolation=t.InterpolationMode.BICUBIC)

        image_color = resize_fn(image_color)
        return image_color

    def preprocessing_fn_semantic(self, image_semantic_label):
        image_semantic_label = torch.from_numpy(image_semantic_label[np.newaxis, :, :])

        resize_fn = t.Resize([self.h_for_inference, self.w_for_inference], interpolation=t.InterpolationMode.NEAREST)

        image_semantic_label = torch.squeeze(resize_fn(image_semantic_label))
        return image_semantic_label

    def postprocessing_fn(self, image):
        image_resized = cv2.resize(image, [self.w, self.h], interpolation=cv2.INTER_NEAREST)
        return image_resized
