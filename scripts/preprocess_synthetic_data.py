import argparse

import numpy as np

from segmentation.utils.data_io import DataIO

from segmentation.modules.synthetic_data.augmenter_synthetic_data import AugmenterSyntheticData
from segmentation.modules.synthetic_data.preprocessor_synthetic_data import PreprocessorSyntheticData


DICT_INDEX_VALID_DEVICE = {
        'color':            [1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
        'depth':            [1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
        'semantic_label':   [1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
        'instance_label':   [1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
        'pose':             [1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
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

COORDINATE_TRANSFORM = np.array([[1,  0, 0, 0],
                                 [0,  0, 1, 0],
                                 [0, -1, 0, 0],
                                 [0,  0, 0, 1]], dtype=np.float64)


def get_parser():
    parser = argparse.ArgumentParser('Preprocessor')
    parser.add_argument('--count-device', '-c', type=int, default=16)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--augmentation', action='store_true')

    # 입력 루트
    parser.add_argument('--input-root', '-i', type=str, default='Z:/datasets/nuclear_waste/20211124')

    # 출력 루트 및 출력 포맷
    parser.add_argument('--output-root', '-o', type=str, default='data_root/synthetic')
    parser.add_argument('--format-frame', '-f', type=str, default='frame_')

    # 디텍터리명 양식
    parser.add_argument('--keyword-color-dir', type=str, default='RGB*')
    parser.add_argument('--keyword-depth-dir', type=str, default='ScreenCapture*')
    parser.add_argument('--keyword-semantic-dir', type=str, default='SemanticSegmentation*')
    parser.add_argument('--keyword-instance-dir', type=str, default='InstanceSegmentation*')
    parser.add_argument('--keyword-camera-dir', type=str, default= 'Dataset*')

    # 이미지파일명 양식
    parser.add_argument('--keyword-color-image', type=str, default='rgb_*.png')
    parser.add_argument('--keyword-depth-image', type=str, default='Depth Camera_depth_*.raw')
    parser.add_argument('--keyword-semantic-image', type=str, default='segmentation_*.png')
    parser.add_argument('--keyword-instance-image', type=str, default='instance_*.png')
    parser.add_argument('--keyword-camera-parameter', type=str, default='captures_*')

    #################################################################################
    # Augmentation Parameter들.
    # 전처리 단계에서 Augmentation 안 할 거면 신경 안 써도 됨.
    parser.add_argument('--mean-distortion-global', type=float, default=0)
    parser.add_argument('--std-distortion-global', type=float, default=500)
    parser.add_argument('--mean-std-gaussian-global', type=float, default=60000)
    parser.add_argument('--std-std-gaussian-global', type=float, default=10000)
    parser.add_argument('--mean-distortion-local', type=float, default=0)
    parser.add_argument('--std-distortion-local', type=float, default=130)
    parser.add_argument('--mean-std-gaussian-local', type=float, default=2000)
    parser.add_argument('--std-std-gaussian-local', type=float, default=1000)
    return parser
    #################################################################################


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    args = vars(args)

    dataio_ = DataIO(args['output_root'], args['format_frame'],
                     count_device=args['count_device'], class_color_map=CLASS_COLOR_MAP)
    if args['augmentation']:
        augmenter = AugmenterSyntheticData(mean_distortion_global=0, std_distortion_global=500,
                                           mean_std_gaussian_global=60000, std_std_gaussian_global=10000,
                                           mean_distortion_local=0, std_distortion_local=130,
                                           mean_std_gaussian_local=2000, std_std_gaussian_local=1000)
    else:
        augmenter = None

    preprocessor = PreprocessorSyntheticData(args['input_root'],
                                             args['keyword_color_dir'],
                                             args['keyword_depth_dir'],
                                             args['keyword_semantic_dir'],
                                             args['keyword_instance_dir'],
                                             args['keyword_camera_dir'],
                                             args['keyword_color_image'],
                                             args['keyword_depth_image'],
                                             args['keyword_semantic_image'],
                                             args['keyword_instance_image'],
                                             args['keyword_camera_parameter'],
                                             DICT_INDEX_VALID_DEVICE, dataio_,
                                             h=args['height'], w=args['width'],
                                             augmenter=augmenter)
    preprocessor.run()


if __name__ == '__main__':
    main()
