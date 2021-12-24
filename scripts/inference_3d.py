import argparse

import cv2
import numpy as np
import pptk

from segmentation.modules.geometry.transformer_geometry import TransformerGeometry
from segmentation.modules.segmentation_network.module_inference import ModuleInference
from segmentation.utils.data_io import  DataIO
from segmentation.utils.transform_extrinsic_representation import rodrigues2extrinsic


CLASS_COLOR_MAP = {
    0: [255, 0, 0],  # wall
    1: [0, 0, 255],  # floor
    2: [0, 255, 0],  # barrel
    3: [255, 255, 255],  # palette
    4: [255, 0, 255],  # forklift
    5: [255, 255, 0],  # person
    6: [0, 255, 255],  # other
    255: [0, 0, 0]  # unknown
}


def get_args():
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--backbone', '-bb', type=str, default='efficientnet-b5')
    parser.add_argument('--data-root', type=str, default='data-root')
    parser.add_argument('--format-frame', type=str, default='frame_for_lion')
    parser.add_argument('--count-device', type=int, default=6)
    parser.add_argument('--param-root', default=None)
    parser.add_argument('--extrinsic', type=str, default='ir_finetuned')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--depth-trunc', type=float, default=6.0)
    parser.add_argument('--path-checkpoint', type=str)
    parser.add_argument('--count-class', type=int)
    parser.add_argument('--show-2d', action='store_true')
    return parser.parse_args()


def map_label2color(arr_label):
    arr_color = np.empty([arr_label.shape[0], arr_label.shape[1], 3], dtype=np.uint8)
    for label, color in CLASS_COLOR_MAP.items():
        arr_color[arr_label == label] = color
    return arr_color


def inference_image(inferencer_, image_color_):
    image_semantic_label_ = inferencer_.inference(image_color_)
    image_semantic_label_colored = map_label2color(image_semantic_label_)
    return image_semantic_label_colored


if __name__ == '__main__':
    args = get_args()
    args = vars(args)

    data_root = args['data_root']
    format_frame = args['format_frame']
    count_device = args['count_device']
    param_root = args['param_root']
    type_extrinsic = args['extrinsic']
    depth_trunc = args['depth_trunc']
    path_checkpoint = args['path_checkpoint']
    count_class = args['count_class']
    show_2d = args['show_2d']

    inferencer = ModuleInference(path_checkpoint, count_class, backbone=args['backbone'])
    # inferencer = ModuleInference(path_checkpoint, count_class)
    dataio = DataIO(root_data=data_root,
                    format_frame=format_frame,
                    count_device=count_device,
                    root_param=param_root)
    coordinate_transformer = TransformerGeometry()

    count_frame_of_camera0 = dataio.count_total_frames(0)
    for num_frame in range(count_frame_of_camera0):
        flag_invalid_frame = False

        print("Loading Data...")
        list_xyzrgbrgb_world_coordinate = []

        for num_device in range(count_device):
            image_color = dataio.load_color(num_device, num_frame)[:, :, :3][:, :, ::-1]
            image_depth = dataio.load_depth(num_device, num_frame)
            image_semantic_label = inference_image(inferencer, image_color)

            if show_2d:
                key = 0
                while True:
                    while key != 27:
                        cv2.imshow('', image_color[:, :, ::-1])
                        key = cv2.waitKey(0)
                        if key == 32 or key == 27:
                            break
                    while key != 27:
                        cv2.imshow('', image_semantic_label[:, :, ::-1])
                        cv2.waitKey(0)
                        if key == 32 or key == 27:
                            break
                    if key == 27:
                        cv2.destroyWindow('')
                        break

            h_color, w_color, _ = image_color.shape
            h_depth, w_depth = image_depth.shape
            intrinsic_color = dataio.load_intrinsic(num_device, 'color')
            intrinsic_depth = dataio.load_intrinsic(num_device, 'ir')
            distortion_color = dataio.load_distortion(num_device, 'color')
            distortion_depth = dataio.load_distortion(num_device, 'ir')
            if not dataio.check_this_frame_storing_specific_files(num_device, 0,
                                                                  ['vec_rotation_{}.txt'.format(type_extrinsic),
                                                                   'vec_translation_{}.txt'.format(type_extrinsic)]):
                flag_invalid_frame = True
                break

            vec_rotation_depth = dataio.load_rotation(num_device, num_frame, type_extrinsic)
            vec_translation_depth = dataio.load_translation(num_device, num_frame, type_extrinsic)
            vec_translation_depth /= 1000
            extrinsic_depth = rodrigues2extrinsic(vec_rotation_depth, vec_translation_depth)
            image_color_registered, image_depth_undistorted, image_semantic_label_registered = image_color, image_depth, image_semantic_label

            coordinate_transformer.initialize_camera_parameter(w_depth, h_depth, intrinsic_depth, extrinsic_depth)
            xyz_world_coordinate = \
                coordinate_transformer.convert_depth_to_point_cloud(image_depth_undistorted)
            rgb = image_color_registered.reshape(-1, 3)
            rgb_label = image_semantic_label_registered.reshape(-1, 3)

            xyzrgbrgb_world_coordinate = np.concatenate([xyz_world_coordinate, rgb, rgb_label], axis=1)
            list_xyzrgbrgb_world_coordinate.append(xyzrgbrgb_world_coordinate)
        if flag_invalid_frame:
            continue
        xyzrgbrgb_world_coordinate_total = np.concatenate(list_xyzrgbrgb_world_coordinate, axis=0)

        viewer = pptk.viewer(xyzrgbrgb_world_coordinate_total[:, :3])
        viewer.attributes(xyzrgbrgb_world_coordinate_total[:, 3:6] / 255, xyzrgbrgb_world_coordinate_total[:, 6:] / 255)
