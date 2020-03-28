import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--gpu', default="0")
arg('--epochs', type=int, default=5)
arg('--fold', default='0')
arg('--n_folds', type=int, default=4)
arg('--freeze_till_layer', default='input_1')
arg('--preprocessing_function', default='caffe')
arg('--weights')
arg('--learning_rate', type=float, default=0.001)
arg('--crop_size', type=int, default=192)
arg('--crops_per_image', type=int, default=1)
arg('--batch_size', type=int, default=11)
arg('--num_workers', type=int, default=7)
arg('--loss_function', default='bce_dice')
arg('--optimizer', default="rmsprop")
arg('--clr')
arg('--schedule')
arg('--decay', type=float, default=0.0)
arg('--save_period', type=int, default=1)
arg('--network', default='densenet_unet')
arg('--alias', default='')
arg('--steps_per_epoch', type=int, default=0)
arg('--use_softmax', action="store_true")
arg('--use_full_masks', action="store_true")
arg('--multi_gpu', action="store_true")
arg('--seed', type=int, default=777)
arg('--models_dir', default='nn_models')
arg('--images_dir', default='../data/images_all')
arg('--labels_dir', default='../data/labels_all')
arg('--test_folder', default='../data_test')
arg('--folds_csv', default='../data/folds.csv')
arg('--out_root_dir', default='../predictions')
arg('--out_masks_folder')
arg('--models',  nargs='+')
arg('--out_channels',  type=int, default=2)

args = parser.parse_args()
