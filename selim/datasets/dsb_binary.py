import random
import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from skimage.filters import median
from skimage.morphology import dilation, watershed, square, erosion
from tqdm import tqdm

from datasets.base import BaseMaskDatasetIterator
# from params import args


class DSB2018BinaryDataset:
    def __init__(self,
                 images_dir,
                 masks_dir,
                 labels_dir,
                 fold=0,
                 fold_num=4,
                 seed=777,
                 ):
        super().__init__()
        self.fold = fold
        self.fold_num = fold_num
        self.seed = seed
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels_dir = labels_dir
        np.random.seed(seed)

    def get_generator(self, crop_shape, preprocessing_function='torch', random_transformer=None, batch_size=16, shuffle=True):
        return DSB2018BinaryDatasetIterator(
            self.images_dir,
            self.masks_dir,
            self.labels_dir,
            crop_shape,
            preprocessing_function,
            random_transformer,
            batch_size,
            shuffle=shuffle,
            image_name_template="{id}.png",
            mask_template="{id}.png",
            label_template="{id}.tif",
            padding=32,
            seed=self.seed
        )

    def train_generator(self, crop_shape=(256, 256), preprocessing_function='torch', random_transformer=None, batch_size=16):
        return self.get_generator(crop_shape, preprocessing_function, random_transformer, batch_size, True)

    def val_generator(self, preprocessing_function='torch', batch_size=1):
        return self.get_generator(None, preprocessing_function, None, batch_size, False)



class DSB2018BinaryDatasetIterator(BaseMaskDatasetIterator):

    def __init__(self, images_dir, masks_dir, labels_dir, crop_shape, preprocessing_function, random_transformer=None, batch_size=8, shuffle=True,
                 image_name_template=None, mask_template=None, label_template=None, padding=32, seed=None):
        super().__init__(images_dir, masks_dir, labels_dir, crop_shape, preprocessing_function, random_transformer, batch_size, shuffle, image_name_template,
                         mask_template, label_template, padding, seed, grayscale_mask=False)

    def transform_mask(self, mask, image):
        mask[mask > 127] = 255

        mask[..., 2] = 255 - mask[...,1]- mask[...,0]
        mask = np.clip(mask, 0, 255)

        return np.array(mask, "float32") / 255.

    def augment_and_crop_mask_image(self, mask, image, label, img_id, crop_shape):
        return self.copy_cells(mask, image, label, img_id, crop_shape)

    def copy_cells(self, mask, image, label, img_id, input_shape):
        img0 = image.copy()
        msk0 = mask.copy()
        lbl0 = label.copy()
        yp = 0
        xp = 0
        #todo: refactor it, copied from Victor's code as is, random crops should be outside of this method
        if img0.shape[0] < input_shape[0]:
            yp = input_shape[0] - img0.shape[0]
        if img0.shape[1] < input_shape[1]:
            xp = input_shape[1] - img0.shape[1]
        if xp > 0 or yp > 0:
            img0 = np.pad(img0, ((0, yp), (0, xp), (0, 0)), 'constant')
            msk0 = np.pad(msk0, ((0, yp), (0, xp), (0, 0)), 'constant')
            lbl0 = np.pad(lbl0, ((0, yp), (0, xp)), 'constant')

        good4copy = self.all_good4copy[img_id]

        x0 = random.randint(0, img0.shape[1] - input_shape[1])
        y0 = random.randint(0, img0.shape[0] - input_shape[0])
        img = img0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
        msk = msk0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
        lbl = lbl0[y0:y0 + input_shape[0], x0:x0 + input_shape[1]]

        if len(good4copy) > 0 and random.random() < 0.05:
            num_copy = random.randrange(1, min(6, len(good4copy) + 1))
            lbl_max = lbl0.max()
            for i in range(num_copy):
                lbl_max += 1
                l_id = random.choice(good4copy)
                lbl_msk = label == l_id
                y1, x1 = np.min(np.where(lbl_msk), axis=1)
                y2, x2 = np.max(np.where(lbl_msk), axis=1)
                lbl_msk = lbl_msk[y1:y2 + 1, x1:x2 + 1]
                lbl_img = img0[y1:y2 + 1, x1:x2 + 1, :]
                if random.random() > 0.5:
                    lbl_msk = lbl_msk[:, ::-1, ...]
                    lbl_img = lbl_img[:, ::-1, ...]
                rot = random.randrange(4)
                if rot > 0:
                    lbl_msk = np.rot90(lbl_msk, k=rot)
                    lbl_img = np.rot90(lbl_img, k=rot)
                x1 = random.randint(max(0, x0 - lbl_msk.shape[1] // 2),
                                    min(img0.shape[1] - lbl_msk.shape[1], x0 + input_shape[1] - lbl_msk.shape[1] // 2))
                y1 = random.randint(max(0, y0 - lbl_msk.shape[0] // 2),
                                    min(img0.shape[0] - lbl_msk.shape[0], y0 + input_shape[0] - lbl_msk.shape[0] // 2))
                tmp = erosion(lbl_msk, square(5))
                lbl_msk_dif = lbl_msk ^ tmp
                tmp = dilation(lbl_msk, square(5))
                lbl_msk_dif = lbl_msk_dif | (tmp ^ lbl_msk)
                lbl0[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]][lbl_msk] = lbl_max
                img0[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]][lbl_msk] = lbl_img[lbl_msk]
                full_diff_mask = np.zeros_like(img0[..., 0], dtype='bool')
                full_diff_mask[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]] = lbl_msk_dif
                img0[..., 0][full_diff_mask] = median(img0[..., 0], mask=full_diff_mask)[full_diff_mask]
                img0[..., 1][full_diff_mask] = median(img0[..., 1], mask=full_diff_mask)[full_diff_mask]
                img0[..., 2][full_diff_mask] = median(img0[..., 2], mask=full_diff_mask)[full_diff_mask]
            img = img0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
            lbl = lbl0[y0:y0 + input_shape[0], x0:x0 + input_shape[1]]
            msk = self.create_mask(lbl)
        return msk, img, lbl

    def create_mask(self,  labels):
        labels = measure.label(labels, neighbors=8, background=0)
        tmp = dilation(labels > 0, square(9))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = dilation(tmp, square(7))
        msk = (255 * tmp).astype('uint8')

        props = measure.regionprops(labels)
        msk0 = 255 * (labels > 0)
        msk0 = msk0.astype('uint8')

        msk1 = np.zeros_like(labels, dtype='bool')

        max_area = np.max([p.area for p in props])

        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    if max_area > 4000:
                        sz = 6
                    else:
                        sz = 3
                else:
                    sz = 3
                    if props[labels[y0, x0] - 1].area < 300:
                        sz = 1
                    elif props[labels[y0, x0] - 1].area < 2000:
                        sz = 2
                uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                                 max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
                    msk0[y0, x0] = 0

        msk1 = 255 * msk1
        msk1 = msk1.astype('uint8')

        msk2 = np.zeros_like(labels, dtype='uint8')
        msk = np.stack((msk0, msk1, msk2))
        msk = np.rollaxis(msk, 0, 3)
        return msk


def test():
    iterator = DSB2018BinaryDataset('imdir', 'mskdir', 'lbldir')