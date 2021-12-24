import os
import glob
import cv2
import numpy as np


class DataIO:
    def __init__(self, root_data, format_frame,
                 count_device=6, root_param=None, class_color_map=None):
        self.root_data = root_data
        self.format_frame = format_frame
        self.count_device = count_device
        for num_device in range(self.count_device):
            path_camera = os.path.join(self.root_data, 'camera{}'.format(num_device))
            os.makedirs(path_camera, exist_ok=True)

        self.root_param = root_data
        self.set_root_param(root_param)

        if class_color_map is None:
            self.class_color_map = {
                0: [255, 0, 0],  # wall
                1: [0, 0, 255],  # floor
                2: [0, 255, 0],  # barrel
                3: [255, 255, 255],  # palette
                4: [255, 0, 255],  # forklift
                5: [255, 255, 0],  # person
                6: [0, 255, 255],  # other
                255: [0, 0, 0]  # unknown
            }
        else:
            self.class_color_map = class_color_map

    def set_root_param(self, root_param):
        if root_param:
            self.root_param = root_param
            for num_device in range(self.count_device):
                os.makedirs(os.path.join(self.root_param, 'camera{}'.format(num_device)), exist_ok=True)

    def save_color(self, num_device, num_frame, arr_color):
        path_to_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)
        # 이미지 파일은 가시성있게 저장.
        cv2.imwrite(os.path.join(path_to_frame, "color.png"), arr_color)
        # Numpy Array는 원본으로 저장 Color : uint8, Ir : float32, Depth : uint16, Semantic_Label : uint8
        np.save(os.path.join(path_to_frame, "color.npy"), arr_color)

    def load_color(self, num_device, num_frame):
        path_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                  '{0}{1:05d}'.format(self.format_frame, num_frame), 'color.npy')
        arr_color = np.load(path_frame)
        return arr_color

    def get_path_color(self, num_device, num_frame):
        frame_dir = glob.glob(
            os.path.join(self.root_data, 'camera{}'.format(num_device), '{}0*'.format(self.format_frame)))[num_frame]
        return os.path.join(frame_dir, 'color.npy')

    def save_ir(self, num_device, num_frame, arr_ir):
        path_to_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)
        # 이미지 파일은 가시성있게 저장.
        cv2.imwrite(os.path.join(path_to_frame, "ir.png"), arr_ir / 65535. * 255)
        # Numpy Array는 원본으로 저장 Color : uint8, Ir : float32, Depth : uint16, Semantic_Label : uint8
        np.save(os.path.join(path_to_frame, "ir.npy"), arr_ir)

    def load_ir(self, num_device, num_frame):
        path_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                  '{0}{1:05d}'.format(self.format_frame, num_frame), 'ir.npy')
        arr_ir = np.load(path_frame)
        return arr_ir

    def get_path_ir(self, num_device, num_frame):
        frame_dir = glob.glob(
            os.path.join(self.root_data, 'camera{}'.format(num_device), '{}0*'.format(self.format_frame)))[num_frame]
        return os.path.join(frame_dir, 'ir.npy')

    def save_depth(self, num_device, num_frame, arr_depth):
        path_to_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)
        # 이미지 파일은 가시성있게 저장.
        cv2.imwrite(os.path.join(path_to_frame, "depth.png"), np.clip(arr_depth / 6000, 0, 1) * 255)
        # Numpy Array는 원본으로 저장 Color : uint8, Ir : float32, Depth : uint16, Semantic_Label : uint8
        np.save(os.path.join(path_to_frame, "depth.npy"), arr_depth)

    def load_depth(self, num_device, num_frame):
        path_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                  '{0}{1:05d}'.format(self.format_frame, num_frame), 'depth.npy')
        arr_depth = np.load(path_frame)
        return arr_depth

    def save_semantic_label(self, num_device, num_frame, arr_semantic_label):
        path_to_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)

        if len(arr_semantic_label.shape) == 3:  # Colorized Label Case
            # 이미지 파일은 가시성있게 Colorized Label로 저장.
            cv2.imwrite(os.path.join(path_to_frame, "semantic_label.png"), arr_semantic_label)
            # Numpy Array는 Class Label로 저장 Color : uint8, Ir : float32, Depth : uint16, Semantic_Label : uint8
            np.save(os.path.join(path_to_frame, "semantic_label.npy"), self.map_color2label(arr_semantic_label[:, :, ::-1]))
        elif len(arr_semantic_label.shape) == 2:  # Class Label Case
            # 이미지 파일은 가시성있게 Class Color로 매핑해서 저장.
            cv2.imwrite(os.path.join(path_to_frame, "semantic_label.png"), self.map_label2color(arr_semantic_label)[:, :, ::-1])
            # Numpy Array는 Label로 저장 Color : uint8, Ir : float32, Depth : uint16, Semantic_Label : uint8
            np.save(os.path.join(path_to_frame, "semantic_label.npy"), arr_semantic_label)
        else:
            print('Invalid Image Shape!')
            exit(-1)

    def load_semantic_label(self, num_device, num_frame, label_unknown_as_other=False):
        path_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                  '{0}{1:05d}'.format(self.format_frame, num_frame), 'semantic_label.npy')
        if os.path.exists(path_frame):
            arr_semantic_label = np.load(path_frame)
            if label_unknown_as_other:
                arr_semantic_label[arr_semantic_label == 255] = 6
            return arr_semantic_label
        else:
            return None

    ###################################################################################################################
    ####################################################### 미완성 상태 #################################################
    ####### save_instance_label() : 현재 이미지만 그대로 보내는 상태. 클래스 레이블을 넘파이 어레이로 보내는 부분 구현 안 됨. #######
    ####### load_instance_label() : 완전 미구현. ########################################################################
    ###################################################################################################################
    def save_instance_label(self, num_device, num_frame, arr_instance_label):
        path_to_frame = os.path.join(self.root_data, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)
        cv2.imwrite(os.path.join(path_to_frame, "instance_label.png"), arr_instance_label)

    def load_instance_label(self, num_device, num_frame):
        print('===========================미구현==========================')
        return '미구현'
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################

    def save_intrinsic(self, num_device, mat_intrinsic, kind_camera):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device), "mat_intrinsic_{}.txt".format(kind_camera))
        np.savetxt(path, mat_intrinsic, fmt='%.18e', delimiter=' ', newline='\n')

    def load_intrinsic(self, num_device, kind_camera):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device), "mat_intrinsic_{}.txt".format(kind_camera))
        mat_intrinsic = np.loadtxt(path)
        return mat_intrinsic

    def save_distortion(self, num_device, vec_distortion, kind_camera):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device), "vec_distortion_{}.txt".format(kind_camera))
        np.savetxt(path, vec_distortion, fmt='%.18e', delimiter=' ', newline='\n')

    def load_distortion(self, num_device, kind_camera):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device), "vec_distortion_{}.txt".format(kind_camera))
        vec_distortion = np.loadtxt(path)
        return vec_distortion

    def save_rotation(self, num_device, num_frame, vec_rotation, kind_camera):
        path_to_frame = os.path.join(self.root_param, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)
        path_to_data = os.path.join(path_to_frame, 'vec_rotation_{}.txt'.format(kind_camera))
        np.savetxt(path_to_data, vec_rotation, fmt='%.18e', delimiter=' ', newline='\n')

    def load_rotation(self, num_device, num_frame, kind_camera):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device),
                            '{0}{1:05d}'.format(self.format_frame, num_frame), 'vec_rotation_{}.txt'.format(kind_camera))
        if os.path.isfile(path):
            vec_rotation = np.loadtxt(path)
            return vec_rotation

    def save_translation(self, num_device, num_frame, vec_translation, kind_camera):
        path_to_frame = os.path.join(self.root_param, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        os.makedirs(path_to_frame, exist_ok=True)
        path_to_data = os.path.join(path_to_frame, 'vec_translation_{}.txt'.format(kind_camera))
        np.savetxt(path_to_data, vec_translation, fmt='%.18e', delimiter=' ', newline='\n')

    def load_translation(self, num_device, num_frame, kind_camera):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device),
                            '{0}{1:05d}'.format(self.format_frame, num_frame),
                            'vec_translation_{}.txt'.format(kind_camera))
        if os.path.isfile(path):
            vec__translation = np.loadtxt(path)
            return vec__translation

    def save_transformation_color_to_depth(self, num_device, transformation_color_to_depth):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device), 'transformation_color_to_depth.txt')
        np.savetxt(path, transformation_color_to_depth, fmt='%.18e', delimiter=' ', newline='\n')

    def load_transformation_color_to_depth(self, num_device):
        path = os.path.join(self.root_param, 'camera{}'.format(num_device), 'transformation_color_to_depth.txt')
        transformation_color_to_depth = np.loadtxt(path)
        return transformation_color_to_depth

    def save_camera_parameters(self, num_device, mat_intrinsic, vec_distortion,
                               list_vec_rotation, list_vec_translation, index_of_valid_frames, kind_camera):
        for idx, num_frame in enumerate(index_of_valid_frames):
            if idx == 0:
                self.save_intrinsic(num_device, mat_intrinsic, kind_camera)
                self.save_distortion(num_device, vec_distortion, kind_camera)
            self.save_rotation_translation(num_device, num_frame, list_vec_rotation[idx], list_vec_translation[idx], kind_camera)

    def load_camera_parameters(self):
        pass

    def save_rotation_translation(self, num_device, num_frame, vec_rotation, vec_translation, kind_camera):
        self.save_rotation(num_device, num_frame, vec_rotation, kind_camera)
        self.save_translation(num_device, num_frame, vec_translation, kind_camera)

    def count_total_frames(self, num_device):
        return len(
                glob.glob(
                    os.path.join(self.root_data, 'camera{}'.format(num_device), '{}0*'.format(self.format_frame))))

    def count_last_frame(self, num_device):
        list_path = glob.glob(
            os.path.join(self.root_data, 'camera{}'.format(num_device), '{}0*'.format(self.format_frame)))
        if list_path:
            max_num_frame = -1
            for path in list_path:
                num_frame = int(path.split('{}0'.format(self.format_frame))[-1])
                max_num_frame = max(num_frame, max_num_frame)
            return max_num_frame
        else:
            print("No frame exists!")
            return -1

    def check_this_frame_storing_specific_files(self, num_device: int, num_frame: int, list_data: list):
        path_to_frame = os.path.join(self.root_param, 'camera{}'.format(num_device),
                                     '{0}{1:05d}'.format(self.format_frame, num_frame))
        # 존재성 감사(프레임)
        if not os.path.exists(path_to_frame):
            return False

        for data in list_data:
            path_to_data = os.path.join(path_to_frame, data)
            # 존재성 검사(데이터)
            if not os.path.exists(path_to_data):
                return False
            # 유효성 검사
            if data in ['vec_rotation_color.txt',
                        'vec_translation_color.txt',
                        'vec_rotation_ir.txt',
                        'vec_translation_color.txt']:
                if np.any(np.loadtxt(path_to_data) == np.array([None, None, None])):
                    return False
        return True

    def get_index_of_frames_storing_specific_files(self, num_device: int, list_data: list):
        num_frame_total = self.count_last_frame(num_device)  # 빠진 프레임이 있어도 마지막 프레임 번호를 알기 위해
        list_frame_valid = []
        for num_frame in range(num_frame_total + 1):
            if self.check_this_frame_storing_specific_files(num_device, num_frame, list_data):
                list_frame_valid.append(num_frame)
        return list_frame_valid

    def map_label2color(self, arr_label):
        arr_color = np.empty([arr_label.shape[0], arr_label.shape[1], 3], dtype=np.uint8)
        for label, color in self.class_color_map.items():
            arr_color[arr_label == label] = color
        return arr_color

    def map_color2label(self, arr_color):
        arr_label = np.empty([arr_color.shape[0], arr_color.shape[1]], dtype=np.uint8)
        for label, color in self.class_color_map.items():
            arr_label[np.all(arr_color == color, axis=2)] = label
        return arr_label
