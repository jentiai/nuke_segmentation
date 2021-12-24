import numpy as np
import torch

from segmentation.modules.segmentation_network.module_model import get_model
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class ModuleInference:
    def __init__(self, path_checkpoint, count_class, backbone="efficientnet-b5"):
        self.count_class = count_class

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'device' in Pytorch Context = Processing Unit
        self.model = get_model(count_class, backbone)
        self.model.load_state_dict(torch.load(path_checkpoint))
        self.model.to(self.device)
        self.model.eval()

        self.preprocessing_fn = get_preprocessing_fn(backbone, pretrained="imagenet")

    def preprocessing_fn_color(self, image_color, w_for_inference, h_for_inference):
        image_color = self.preprocessing_fn(image_color)
        image_color = torch.from_numpy(np.transpose(image_color, [2, 0, 1])[np.newaxis, :, :, :]).float()  # [H, W, C] => [1, C, H, W]
        # resize_fn = t.Resize([h_for_inference, w_for_inference], interpolation=t.InterpolationMode.BICUBIC)
        # image_color = resize_fn(image_color)
        return image_color

    def postprocessing_fn_semantic_label(self, batch_prediction, w_origin, h_origin):
        image_semantic_label = torch.argmax(batch_prediction, dim=1).detach().cpu().numpy()[0]  # [1, C, H, W] => [H, W]
        # image_semantic_label = cv2.resize(image_semantic_label, [w_origin, h_origin], interpolation=cv2.INTER_NEAREST)
        return image_semantic_label

    @torch.no_grad()
    def inference(self, image_color):
        h_origin, w_origin, c = image_color.shape
        if c != 3:
            print("Image is not colored!")
            exit(-1)
        ar = w_origin / h_origin
        if w_origin > 640:  # GPU 메모리 부족
            w_resized = 640
            h_resized = int(w_resized / ar)
        else:
            w_resized = w_origin
            h_resized = h_origin

        if h_resized > 640:  # GPU 메모리 부족
            h_resized = 640
            w_resized = int(h_resized * ar)
        else:
            w_resized = w_resized
            h_resized = h_resized

        w_for_inference = w_resized - w_resized % 32
        h_for_inference = h_resized - h_resized % 32

        batch_color = self.preprocessing_fn_color(image_color, w_for_inference, h_for_inference).to(device=self.device, dtype=torch.float32)
        batch_prediction = self.model(batch_color)
        image_semantic_label = self.postprocessing_fn_semantic_label(batch_prediction, w_origin, h_origin)
        return image_semantic_label
