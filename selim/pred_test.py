import os

from params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from models.model_factory import make_model

from datasets.lidc import imread
from os import path, mkdir, listdir
import numpy as np

np.random.seed(1)
import random

random.seed(1)
import tensorflow as tf

tf.set_random_seed(1)
import timeit
import cv2
from tqdm import tqdm

test_folder = args.test_folder
test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)

all_ids = []
all_images = []
all_masks = []

OUT_CHANNELS = args.out_channels

def preprocess_inputs(x):
    return preprocess_input(x, mode=args.preprocessing_function)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    for w in weights:
        model = make_model(args.network, (None, None, 3))
        print("Building model {} from weights {} ".format(args.network, w))
        model.load_weights(w)
        models.append(model)
    os.makedirs(test_pred, exist_ok=True)
    print('Predicting test')
    for d in tqdm(listdir(test_folder)):
        if not path.isdir(path.join(test_folder, d)):
            continue
        img = imread(test_folder + '/' + d)

        for model in models:
            pred = model.predict(img, batch_size=1)

        cv2.imwrite(test_folder + '/' + d, pred)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))