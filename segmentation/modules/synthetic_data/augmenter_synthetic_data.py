import cv2
import numpy as np
from numba import jit


@jit(nopython=True)
def get_gaussian_kernel_jit(w, h, std, center_x=256, center_y=212):
    center_x_window_coordinate = center_x - w // 2
    center_y_window_coordinate = center_y - h // 2
    coord_x = np.linspace((-w // 2) - center_x_window_coordinate, (w // 2) - center_x_window_coordinate, w)
    coord_y = np.linspace((-h // 2) - center_y_window_coordinate, (h // 2) - center_y_window_coordinate, h)
    kernel_gaussian_1d_x = np.exp(-0.5 * np.square(coord_x) / std)
    kernel_gaussian_1d_y = np.exp(-0.5 * np.square(coord_y) / std)
    kernel_gaussian_2d = np.outer(kernel_gaussian_1d_y, kernel_gaussian_1d_x.T)
    return kernel_gaussian_2d / np.sum(kernel_gaussian_2d)


@jit(nopython=True)
def get_global_distortion_kernel_jit(w, h, mean_distortion, std_distortion, std_gaussian):
    kernel_gaussian_2d = get_gaussian_kernel_jit(w, h, std_gaussian)
    kernel_distortion = np.ones_like(kernel_gaussian_2d) * np.max(kernel_gaussian_2d) - kernel_gaussian_2d
    weight_distortion = np.random.normal(mean_distortion, scale=std_distortion)
    return weight_distortion * kernel_distortion / np.max(kernel_distortion)


@jit(nopython=True)
def get_local_distortion_kernel_jit(w, h, mean_distortion, std_distortion, std_gaussian, center_gaussian_x, center_gaussian_y):
    kernel_gaussian_2d = get_gaussian_kernel_jit(w, h, std_gaussian, center_gaussian_x, center_gaussian_y)
    weight_distortion = np.random.normal(mean_distortion, scale=std_distortion)
    # weight_distortion를 std_gaussian에 따라서 일부 조절.
    return weight_distortion * kernel_gaussian_2d / np.max(kernel_gaussian_2d)


@jit(nopython=True)
def add_local_gaussian_blob_jit(image, mean_distortion, std_distortion, mean_std_gaussian, std_std_gaussian):
    # image = np.ones_like(image) * 300
    count_blob = np.random.randint(0, 150)
    for i in range(count_blob):
        w = image.shape[1]
        h = image.shape[0]
        x_blob = np.random.randint(0, w-1)
        y_blob = np.random.randint(0, h-1)
        std_gaussian = np.abs(np.random.normal(mean_std_gaussian, scale=std_std_gaussian))
        kernel_distortion_local = get_local_distortion_kernel_jit(w, h, mean_distortion, std_distortion, std_gaussian, x_blob, y_blob)
        image += kernel_distortion_local
        # cv2.imshow('', image / 600)
        # cv2.waitKey()
    return image


class AugmenterSyntheticData:
    def __init__(self,
                 mean_distortion_global=0, std_distortion_global=500,
                 mean_std_gaussian_global=60000, std_std_gaussian_global=10000,
                 mean_distortion_local=0, std_distortion_local=130,
                 mean_std_gaussian_local=2000, std_std_gaussian_local=1000):
        self.mean_distortion_global = mean_distortion_global
        self.std_distortion_global = std_distortion_global
        self.mean_std_gaussian_global = mean_std_gaussian_global
        self.std_std_gaussian_global = std_std_gaussian_global
        self.mean_distortion_local = mean_distortion_local
        self.std_distortion_local= std_distortion_local
        self.mean_std_gaussian_local = mean_std_gaussian_local
        self.std_std_gaussian_local = std_std_gaussian_local

    def add_local_gaussian_distortion(self, image_depth):
        image_depth_distorted = add_local_gaussian_blob_jit(image_depth, self.mean_distortion_local, self.std_distortion_local, self.mean_std_gaussian_local, self.std_std_gaussian_local)
        return np.clip(image_depth_distorted, 0, np.max(image_depth_distorted))

    def add_global_gaussian_distortion(self, image_depth):
        std_gaussian = np.abs(np.random.normal(self.mean_std_gaussian_global, scale=self.std_std_gaussian_global))
        kernel_distortion_global = get_global_distortion_kernel_jit(image_depth.shape[1], image_depth.shape[0], self.mean_distortion_global, self.std_distortion_global, std_gaussian)
        image_depth_distorted = image_depth + kernel_distortion_global
        return np.clip(image_depth_distorted, 0, np.max(image_depth_distorted))

    # def add_gaussian_blur(self, image_depth):
    #     return cv2.GaussianBlur(image_depth, [3, 3], 0)

    # def add_occluded_area(self, image_depth):
    #     return image_depth, index_invalid

    # def remove_edge(self, image_depth):
    #     return image_depth, index_invalid




# def augment_synthetic_data(image_color, image_depth, image_label, vec_rotation_depth, vec_translation_depth):
#
#     return image_color_augmented, image_depth_augmented, image_label_augmented, vec_rotation_depth_augmented, vec_translation_depth_augmented


if __name__ == '__main__':
    augmenter = AugmenterSyntheticData()


