import argparse

from segmentation.utils.data_io import DataIO
from segmentation.modules.segmentation_network.module_train import ModuleTrain


DICT_NUM2CLASS = {
        0: 'wall',
        1: 'floor',
        2: 'barrel',
        3: 'palette',
        4: 'forklift',
        5: 'person',
        6: 'other'
    }


def get_args():
    parser = argparse.ArgumentParser(description='Train the custom_segmentation_network on images and target masks')
    parser.add_argument('--backbone', '-bb', type=str, default='efficientnet-b5')
    parser.add_argument('--data-root', '-r', type=str, default='data_root/synthetic')
    parser.add_argument('--format-frame', '-f', type=str, default='frame_')
    parser.add_argument('--count-device', '-d', type=int, default=16)
    parser.add_argument('--count-class', '-c', type=int, default=7)
    parser.add_argument('--dir-checkpoint', '-ch', type=str, default='weights/ckpt')
    parser.add_argument('--max-epoch', '-e', type=int, default=200)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--learning-rate', '-l', type=float, default=0.008)
    parser.add_argument('--num-worker', '-w', type=int, default=4)
    parser.add_argument('--freq-save', '-s', type=int, default=10)
    parser.add_argument('--freq-val', '-v', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    dataio = DataIO(args['data_root'], args['format_frame'], count_device=args['count_device'])

    trainer = ModuleTrain(dataio, args['count_class'], args['batch_size'], args['learning_rate'],
                          backbone=args['backbone'],
                          num_worker=args['num_worker'],
                          dict_num2class=DICT_NUM2CLASS)
    trainer.run(args['max_epoch'], args['freq_save'], args['freq_val'], args['dir_checkpoint'])
