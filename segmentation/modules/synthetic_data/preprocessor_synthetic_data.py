import os

import json
from glob import glob
import imageio
import numpy as np

from segmentation.utils.data_io import DataIO
from segmentation.utils.transform_extrinsic_representation import quaternion2rodrigues, rodrigues2extrinsic, extrinsic2rodrigues
from segmentation.modules.synthetic_data.augmenter_synthetic_data import AugmenterSyntheticData


class PreprocessorSyntheticData:
    def __init__(self, root_data,
                 keyword_color_dir,
                 keyword_depth_dir,
                 keyword_semantic_label_dir,
                 keyword_instance_label_dir,
                 keyword_camera_dir,
                 keyword_color_image,
                 keyword_depth_image,
                 keyword_semantic_label_image,
                 keyword_instance_label_image,
                 keyword_camera_parameter,
                 dict_index_valid_device, dataio, h=424, w=512, coordinate_transform=None, augmenter=None):
        self.root_data = root_data
        self.list_scene = os.listdir(root_data)

        self.keyword_color_dir = keyword_color_dir
        self.keyword_depth_dir = keyword_depth_dir
        self.keyword_semantic_label_dir = keyword_semantic_label_dir
        self.keyword_instance_label_dir = keyword_instance_label_dir
        self.keyword_camera_dir = keyword_camera_dir

        self.keyword_color_image = keyword_color_image
        self.keyword_depth_image = keyword_depth_image
        self.keyword_semantic_label_image = keyword_semantic_label_image
        self.keyword_instance_label_image = keyword_instance_label_image
        self.keyword_camera_parameter = keyword_camera_parameter

        self.dataio = dataio
        self.h = h
        self.w = w
        self.augmenter = augmenter

        if coordinate_transform is None:
            self.coordinate_transform = np.array([[1, 0, 0, 0],
                                                  [0, 0, 1, 0],
                                                  [0, -1, 0, 0],
                                                  [0, 0, 0, 1]], dtype=np.float64)
        else:
            self.coordinate_transform = coordinate_transform

        if len(dict_index_valid_device['color']) == len(dict_index_valid_device['depth']) ==\
                len(dict_index_valid_device['semantic_label']) == len(dict_index_valid_device['instance_label']) == \
                len(dict_index_valid_device['pose']):

            self.list_index_valid_device_color = dict_index_valid_device['color']
            self.list_index_valid_device_depth = dict_index_valid_device['depth']
            self.list_index_valid_device_semantic_label = dict_index_valid_device['semantic_label']
            self.list_index_valid_device_instance_label = dict_index_valid_device['instance_label']
            self.list_index_valid_device_pose = dict_index_valid_device['pose']
        else:
            print('Lengths of valid device between frames are not same!')
            exit(-1)

    def get_data_path(self, num_frame):
        path_frame = os.path.join(self.root_data, self.list_scene[num_frame])
        dir_color = glob(os.path.join(path_frame, self.keyword_color_dir))[0]
        dir_depth = glob(os.path.join(path_frame, self.keyword_depth_dir))[0]
        dir_semantic_label = glob(os.path.join(path_frame, self.keyword_semantic_label_dir))[0]
        dir_instance_label = glob(os.path.join(path_frame, self.keyword_instance_label_dir))[0]
        dir_camera = glob(os.path.join(path_frame, self.keyword_camera_dir))[0]
        return path_frame, dir_color, dir_depth, dir_semantic_label, dir_instance_label, dir_camera

    def load_color_list_from_dir(self, dir_color):
        list_path_color_image = glob(os.path.join(dir_color, self.keyword_color_image))
        return [imageio.imread(list_path_color_image[index_valid])[:, :, :3][:, :, ::-1]
                for index_valid in self.list_index_valid_device_color]

    def load_depth_list_from_dir(self, dir_depth):
        list_depth = []
        for index_valid in self.list_index_valid_device_depth:
            image_depth = np.fromfile(
            os.path.join(dir_depth, self.keyword_depth_image.replace('*', str(index_valid))),
            dtype=np.uint16).reshape([self.h, self.w])[::-1, :]

            if self.augmenter is not None:
                image_depth = self.augmenter.add_global_gaussian_distortion(image_depth)
                image_depth = self.augmenter.add_local_gaussian_distortion(image_depth)

            list_depth.append(image_depth)
        return list_depth

    def load_semantic_label_list_from_dir(self, dir_semantic_label):
        list_path_semantic_label_image = glob(os.path.join(dir_semantic_label, self.keyword_semantic_label_image))
        return [imageio.imread(list_path_semantic_label_image[index_valid])[:, :, :3][:, :, ::-1]
                for index_valid in self.list_index_valid_device_semantic_label]

    def load_instance_label_list_from_dir(self, dir_instance_label):
        list_path_instance_label_image = glob(os.path.join(dir_instance_label, self.keyword_instance_label_image))
        return [imageio.imread(list_path_instance_label_image[index_valid])[:, :, :3][:, :, ::-1]
                for index_valid in self.list_index_valid_device_instance_label]

    def load_camera_parameter_list_from_dir(self, dir_camera):
        path_camera_parameter = glob(os.path.join(dir_camera, self.keyword_camera_parameter))[0]
        with open(path_camera_parameter) as fd:
            camera_parameter = json.load(fd)['captures']
        return [camera_parameter[index_valid] for index_valid in self.list_index_valid_device_pose]

    def transform_intrinsic(self, intrinsic_unity_perception):
        fx = intrinsic_unity_perception[0][0] * self.w / 2
        fy = intrinsic_unity_perception[1][1] * self.h / 2
        cx = self.w / 2
        cy = self.h / 2
        intrinsic_kinect = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        return intrinsic_kinect

    def transform_extrinsic(self, quat_rotation_unity, vec_translation_unity):
        #  좌표계 변환 : Unity Left Handed => Right Handed
        rotation_quat_kinect = np.array(
            [-quat_rotation_unity[0], quat_rotation_unity[1], -quat_rotation_unity[2], quat_rotation_unity[3]])
        translation_kinect = np.array(
            [[vec_translation_unity[0]], [-vec_translation_unity[1]], [vec_translation_unity[2]]]) * 1000

        #  pose => extrinsic
        extrinsic_kinect = np.linalg.inv(rodrigues2extrinsic(quaternion2rodrigues(rotation_quat_kinect), translation_kinect))

        # 좌표계 변환 : Z축이 위로 가도록
        extrinsic_kinect = extrinsic_kinect @ np.linalg.inv(self.coordinate_transform)
        vec_rotation_kinect, vec_translation_kinect = extrinsic2rodrigues(extrinsic_kinect)
        return vec_rotation_kinect, vec_translation_kinect

    def get_intrinsic_from_list(self, list_camera_parameter):
        list_intrinsic = [self.transform_intrinsic(camera_parameter['sensor']['camera_intrinsic'])
                          for camera_parameter in list_camera_parameter]
        return list_intrinsic

    def get_rotation_translation_from_list(self, list_camera_parameter):
        list_rotation = []
        list_translation = []
        for camera_parameter in list_camera_parameter:
            quat_rotation_unity = np.array(camera_parameter['ego']['rotation'])
            vec_translation_unity = np.array(camera_parameter['ego']['translation'])

            vec_rotation_kinect, vec_translation_kinect = self.transform_extrinsic(quat_rotation_unity, vec_translation_unity)

            list_rotation.append(vec_rotation_kinect)
            list_translation.append(vec_translation_kinect)
        return list_rotation, list_translation

    def save_color_from_list(self, num_frame, list_color):
        for index, color in enumerate(list_color):
            self.dataio.save_color(index, num_frame, color)

    def save_depth_from_list(self, num_frame, list_depth):
        for index, depth in enumerate(list_depth):
            self.dataio.save_depth(index, num_frame, depth)

    def save_semantic_label_from_list(self, num_frame, list_semantic_label):
        for index, semantic_label in enumerate(list_semantic_label):
            self.dataio.save_semantic_label(index, num_frame, semantic_label)

    def save_instance_label_from_list(self, num_frame, list_instance_label):
        for index, instance_label in enumerate(list_instance_label):
            self.dataio.save_instance_label(index, num_frame, instance_label)

    def save_intrinsic_from_list(self, list_camera_parameter):
        list_intrinsic = self.get_intrinsic_from_list(list_camera_parameter)
        for index, intrinsic in enumerate(list_intrinsic):
            # 합성데이터라서 color-depth간 동일좌표계
            self.dataio.save_intrinsic(index, intrinsic, 'ir')
            self.dataio.save_intrinsic(index, intrinsic, 'color')
            # 합성데이터라서 왜곡 없음.
            self.dataio.save_distortion(index, np.array([[0., 0., 0., 0., 0.]], dtype=np.float64), 'ir')
            self.dataio.save_distortion(index, np.array([[0., 0., 0., 0., 0.]], dtype=np.float64), 'color')

    def save_rotation_translation_from_list(self, num_frame, list_camera_parameter):
        list_rotation, list_translation = self.get_rotation_translation_from_list(list_camera_parameter)
        for index, (rotation, translation) in enumerate(zip(list_rotation, list_translation)):
            # 합성데이터라서 color-depth간 동일좌표계
            self.dataio.save_rotation_translation(index, num_frame, rotation, translation, 'ir')
            self.dataio.save_rotation_translation(index, num_frame, rotation, translation, 'color')

    def run(self):
        print('Save Intrinsic Camera Parameter')
        _, _, _, _, _, dir_camera = self.get_data_path(0)
        list_camera_parameter = self.load_camera_parameter_list_from_dir(dir_camera)
        self.save_intrinsic_from_list(list_camera_parameter)

        for num_frame in range(len(self.list_scene)):
            path_frame, dir_color, dir_depth, dir_semantic_label, dir_instance_label, dir_camera = \
                self.get_data_path(num_frame)
            print('\n[SCENE : %s]' % path_frame)

            print('Save Color Image')
            list_color = self.load_color_list_from_dir(dir_color)
            self.save_color_from_list(num_frame, list_color)

            print('Save Depth Image')
            list_depth = self.load_depth_list_from_dir(dir_depth)
            self.save_depth_from_list(num_frame, list_depth)

            print('Save Semantic Label')
            list_semantic_label = self.load_semantic_label_list_from_dir(dir_semantic_label)
            self.save_semantic_label_from_list(num_frame, list_semantic_label)

            print('Save Instance Label')
            list_instance_label = self.load_instance_label_list_from_dir(dir_instance_label)
            self.save_instance_label_from_list(num_frame, list_instance_label)

            print('Save Extrinsic Camera Parameter')
            list_camera_parameter = self.load_camera_parameter_list_from_dir(dir_camera)
            self.save_rotation_translation_from_list(num_frame, list_camera_parameter)


if __name__ == '__main__':
    ROOT_SYNTHETIC_DATA = 'Z:/datasets/nuclear_waste/20210929'
    ROOT_PREPROCESSED_DATA = 'data-root/synthetic'
    FORMAT_FRAME = 'frame_synthetic_01_'
    KEYWORD_COLOR_DIR = 'RGB*'
    KEYWORD_DEPTH_DIR = 'ScreenCapture*'
    KEYWORD_SEMANTIC_LABEL_DIR = 'SemanticSegmentation*'
    KEYWORD_INSTANCE_LABEL_DIR = 'InstanceSegmentation*'
    KEYWORD_CAMERA_DIR = 'Dataset*'
    KEYWORD_COLOR_IMAGE = 'rgb_*.png'
    KEYWORD_DEPTH_IMAGE = 'Depth Camera_depth_*.raw'
    KEYWORD_SEMANTIC_LABEL_IMAGE = 'segmentation_*.png'
    KEYWORD_INSTANCE_LABEL_IMAGE = 'instance_*.png'
    KEYWORD_CAMERA_PARAMETER = 'captures_*'
    COUNT_DEVICE = 15
    FLAG_AUGMENTATION = True
    DICT_INDEX_VALID_DEVICE = {
        'color':            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'depth':            [1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15],
        'semantic_label':   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'instance_label':   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'pose':             [0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14]
    }
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

    dataio_ = DataIO(ROOT_PREPROCESSED_DATA, FORMAT_FRAME,
                     count_device=COUNT_DEVICE, class_color_map=CLASS_COLOR_MAP)

    if FLAG_AUGMENTATION:
        augmenter = AugmenterSyntheticData()
    else:
        augmenter = None

    preprocessor = PreprocessorSyntheticData(ROOT_SYNTHETIC_DATA,
                                             KEYWORD_COLOR_DIR,
                                             KEYWORD_DEPTH_DIR,
                                             KEYWORD_SEMANTIC_LABEL_DIR,
                                             KEYWORD_INSTANCE_LABEL_DIR,
                                             KEYWORD_CAMERA_DIR,
                                             KEYWORD_COLOR_IMAGE,
                                             KEYWORD_DEPTH_IMAGE,
                                             KEYWORD_SEMANTIC_LABEL_IMAGE,
                                             KEYWORD_INSTANCE_LABEL_IMAGE,
                                             KEYWORD_CAMERA_PARAMETER,
                                             DICT_INDEX_VALID_DEVICE, dataio_, augmenter=augmenter)
    preprocessor.run()
