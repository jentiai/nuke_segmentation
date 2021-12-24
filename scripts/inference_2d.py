import argparse

import cv2
import numpy as np

from segmentation.modules.segmentation_network.module_inference import ModuleInference

CLASS_COLOR_MAP = {
    0: [255, 0, 0],  # wall
    1: [0, 0, 255],  # floor
    2: [0, 255, 0],  # barrel
    3: [255, 255, 255],  # palette
    4: [255, 0, 255],  # forklift
    5: [255, 255, 0],  # person
    6: [0, 255, 255],  # other
}


def get_args():
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--path-image', '-i', type=str)
    parser.add_argument('--path-checkpoint', '-ch', type=str)
    parser.add_argument('--count-class', '-c', type=int, default=7)
    return parser.parse_args()


def map_label2color(arr_label):
    arr_color = np.empty([arr_label.shape[0], arr_label.shape[1], 3], dtype=np.uint8)
    for label, color in CLASS_COLOR_MAP.items():
        arr_color[arr_label == label] = color
    return arr_color


if __name__ == '__main__':
    args = get_args()
    args = vars(args)

    inferencer = ModuleInference(args['path_checkpoint'], args['count_class'], backbone="efficientnet-b5")
    image_color = cv2.imread(args['path_image'])
    image_semantic_label = inferencer.inference(image_color)
    image_semantic_label_colored = map_label2color(image_semantic_label)

    key = 0
    while True:
        while key != 27:
            cv2.imshow('Inference Result', image_color)
            key = cv2.waitKey(0)
            if key == 32 or key == 27:
                break
        while key != 27:
            cv2.imshow('Inference Result', image_semantic_label_colored)
            cv2.waitKey(0)
            if key == 32 or key == 27:
                break
        if key == 27:
            cv2.destroyWindow('Inference Result')
            break
