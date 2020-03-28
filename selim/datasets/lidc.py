import numpy as np
from tensorflow.keras.preprocessing.image import Iterator
import time
import os
import xml.etree.ElementTree as ET
import cv2
import pydicom as dicom


def make_mask(image, image_id, nodules):
    height, width, _ = image.shape
    nodule_image = np.zeros((height, width), np.uint8)
    # todo OR for all masks
    edge_map = None
    for nodule in nodules[:1]:
        for roi in nodule['roi']:
            # todo ==
            print(roi['sop_uid'])
            print(image_id)
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                break

    if edge_map is None:
        return image

    # todo what color to fill?
    cv2.fillPoly(nodule_image, np.int32([np.array(edge_map)]), (122, 122, 122))
    mask = cv2.bitwise_and(image, image, mask=nodule_image)
    mask = mask[:,:,:-1]
    print("mask created")
    print(mask.shape)
    return mask


def test():
    nodules = parseXML('/Users/mkryuchkov/lung-ds/3000566-03192')
    image = cv2.imread('/Users/mkryuchkov/lung-ds/000001.jpg')
    image_id = 1
    make_mask(image, image_id, nodules)


def imread(image_path):
    ds = dicom.dcmread(image_path)
    img = ds.pixel_array
    img_2d = img.astype(float)

    ## Step 2. Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    ## Step 3. Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)
    image = img_2d_scaled
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.imwrite('image3d.jpg', image)
    # image3d = cv2.imread('image3d.jpg')
    print(image.shape)
    return image, ds

def kek():
    image3d, ds = imread('/Users/mkryuchkov/Downloads/000500.dcm')
    return image3d

def parseXML(scan_path):
    '''
    parse xml file
    args:
    xml file path
    output:
    nodule list
    [{nodule_id, roi:[{z, sop_uid, xy:[[x1,y1],[x2,y2],...]}]}]
    '''
    file_list = os.listdir(scan_path)
    print
    for file in file_list:
        if '.' in file and file.split('.')[1] == 'xml':
            xml_file = file
            break
    prefix = "{http://www.nih.gov}"
    tree = ET.parse(scan_path + '/' + xml_file)
    root = tree.getroot()
    readingSession_list = root.findall(prefix + "readingSession")
    nodules = []

    for session in readingSession_list:
        # print(session)
        unblinded_list = session.findall(prefix + "unblindedReadNodule")
        for unblinded in unblinded_list:
            nodule_id = unblinded.find(prefix + "noduleID").text
            edgeMap_num = len(unblinded.findall(prefix + "roi/" + prefix + "edgeMap"))
            if edgeMap_num >= 1:
                # it's segmentation label
                nodule_info = {}
                nodule_info['nodule_id'] = nodule_id
                nodule_info['roi'] = []
                roi_list = unblinded.findall(prefix + "roi")
                for roi in roi_list:
                    roi_info = {}
                    # roi_info['z'] = float(roi.find(prefix + "imageZposition").text)
                    roi_info['sop_uid'] = roi.find(prefix + "imageSOP_UID").text
                    roi_info['xy'] = []
                    edgeMap_list = roi.findall(prefix + "edgeMap")
                    for edgeMap in edgeMap_list:
                        x = float(edgeMap.find(prefix + "xCoord").text)
                        y = float(edgeMap.find(prefix + "yCoord").text)
                        xy = [x, y]
                        roi_info['xy'].append(xy)
                    nodule_info['roi'].append(roi_info)
                nodules.append(nodule_info)
    return nodules


class LIDCDatasetIterator(Iterator):
    def __init__(self, image_dir, batch_size):
        seed = np.uint32(time.time() * 1000)
        n = len(os.listdir(image_dir))
        self.image_dir = image_dir
        self.image_ids = self.create_image_ids()
        self.data_shape = (256, 256)
        super().__init__(n, batch_size, False, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        print('len(index_array) : {}'.format(len(index_array)))
        for image_index in index_array:
            file_name, parent_name = self.image_ids[image_index]
            image, dcm_ds = imread(file_name)
            nodules = parseXML(parent_name)
            print('processing image: {}'.format(file_name))
            image = cv2.resize(image, self.data_shape)
            mask = make_mask(image, dcm_ds.get('UID'), nodules)
            mask = cv2.resize(mask, self.data_shape)
            batch_x.append(image)
            batch_y.append(mask)
        batch_x = np.array(batch_x, dtype=np.float64)
        batch_y = np.array(batch_y, dtype=np.float64)
        print("batch_x.shape:")
        print(batch_x.shape)
        print("batch_y.shape:")
        print(batch_y.shape)
        print(self.batch_size)
        return batch_x, batch_y

    def create_image_ids(self):
        dcms = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('dcm'):
                    dcms.append((root + '/' + file, root))
        image_ids = {}
        for i, dcm in enumerate(dcms):
            image_ids[i] = dcm
        return image_ids
    #
    # def transform_mask(self, mask, image):
    #     mask[mask > 127] = 255
    #
    #     #todo: fix args leak
    #     # if not args.use_softmax:
    #     #     mask = mask[..., :2]
    #     # else:
    #     mask[..., 2] = 255 - mask[...,1]- mask[...,0]
    #     mask = np.clip(mask, 0, 255)
    #
    #     return np.array(mask, "float32") / 255.
    #
    # def augment_and_crop_mask_image(self, mask, image, label, img_id, crop_shape):
    #     return self.copy_cells(mask, image, label, img_id, crop_shape)
    #
    # def copy_cells(self, mask, image, label, img_id, input_shape):
    #     img0 = image.copy()
    #     msk0 = mask.copy()
    #     lbl0 = label.copy()
    #     yp = 0
    #     xp = 0
    #     #todo: refactor it, copied from Victor's code as is, random crops should be outside of this method
    #     if img0.shape[0] < input_shape[0]:
    #         yp = input_shape[0] - img0.shape[0]
    #     if img0.shape[1] < input_shape[1]:
    #         xp = input_shape[1] - img0.shape[1]
    #     if xp > 0 or yp > 0:
    #         img0 = np.pad(img0, ((0, yp), (0, xp), (0, 0)), 'constant')
    #         msk0 = np.pad(msk0, ((0, yp), (0, xp), (0, 0)), 'constant')
    #         lbl0 = np.pad(lbl0, ((0, yp), (0, xp)), 'constant')
    #
    #     good4copy = self.all_good4copy[img_id]
    #
    #     x0 = random.randint(0, img0.shape[1] - input_shape[1])
    #     y0 = random.randint(0, img0.shape[0] - input_shape[0])
    #     img = img0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
    #     msk = msk0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
    #     lbl = lbl0[y0:y0 + input_shape[0], x0:x0 + input_shape[1]]
    #
    #     if len(good4copy) > 0 and random.random() < 0.05:
    #         num_copy = random.randrange(1, min(6, len(good4copy) + 1))
    #         lbl_max = lbl0.max()
    #         for i in range(num_copy):
    #             lbl_max += 1
    #             l_id = random.choice(good4copy)
    #             lbl_msk = label == l_id
    #             y1, x1 = np.min(np.where(lbl_msk), axis=1)
    #             y2, x2 = np.max(np.where(lbl_msk), axis=1)
    #             lbl_msk = lbl_msk[y1:y2 + 1, x1:x2 + 1]
    #             lbl_img = img0[y1:y2 + 1, x1:x2 + 1, :]
    #             if random.random() > 0.5:
    #                 lbl_msk = lbl_msk[:, ::-1, ...]
    #                 lbl_img = lbl_img[:, ::-1, ...]
    #             rot = random.randrange(4)
    #             if rot > 0:
    #                 lbl_msk = np.rot90(lbl_msk, k=rot)
    #                 lbl_img = np.rot90(lbl_img, k=rot)
    #             x1 = random.randint(max(0, x0 - lbl_msk.shape[1] // 2),
    #                                 min(img0.shape[1] - lbl_msk.shape[1], x0 + input_shape[1] - lbl_msk.shape[1] // 2))
    #             y1 = random.randint(max(0, y0 - lbl_msk.shape[0] // 2),
    #                                 min(img0.shape[0] - lbl_msk.shape[0], y0 + input_shape[0] - lbl_msk.shape[0] // 2))
    #             tmp = erosion(lbl_msk, square(5))
    #             lbl_msk_dif = lbl_msk ^ tmp
    #             tmp = dilation(lbl_msk, square(5))
    #             lbl_msk_dif = lbl_msk_dif | (tmp ^ lbl_msk)
    #             lbl0[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]][lbl_msk] = lbl_max
    #             img0[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]][lbl_msk] = lbl_img[lbl_msk]
    #             full_diff_mask = np.zeros_like(img0[..., 0], dtype='bool')
    #             full_diff_mask[y1:y1 + lbl_msk.shape[0], x1:x1 + lbl_msk.shape[1]] = lbl_msk_dif
    #             img0[..., 0][full_diff_mask] = median(img0[..., 0], mask=full_diff_mask)[full_diff_mask]
    #             img0[..., 1][full_diff_mask] = median(img0[..., 1], mask=full_diff_mask)[full_diff_mask]
    #             img0[..., 2][full_diff_mask] = median(img0[..., 2], mask=full_diff_mask)[full_diff_mask]
    #         img = img0[y0:y0 + input_shape[0], x0:x0 + input_shape[1], :]
    #         lbl = lbl0[y0:y0 + input_shape[0], x0:x0 + input_shape[1]]
    #         msk = self.create_mask(lbl)
    #     return msk, img, lbl
    #
    # def create_mask(self,  labels):
    #     labels = measure.label(labels, neighbors=8, background=0)
    #     tmp = dilation(labels > 0, square(9))
    #     tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    #     tmp = tmp ^ tmp2
    #     tmp = dilation(tmp, square(7))
    #     msk = (255 * tmp).astype('uint8')
    #
    #     props = measure.regionprops(labels)
    #     msk0 = 255 * (labels > 0)
    #     msk0 = msk0.astype('uint8')
    #
    #     msk1 = np.zeros_like(labels, dtype='bool')
    #
    #     max_area = np.max([p.area for p in props])
    #
    #     for y0 in range(labels.shape[0]):
    #         for x0 in range(labels.shape[1]):
    #             if not tmp[y0, x0]:
    #                 continue
    #             if labels[y0, x0] == 0:
    #                 if max_area > 4000:
    #                     sz = 6
    #                 else:
    #                     sz = 3
    #             else:
    #                 sz = 3
    #                 if props[labels[y0, x0] - 1].area < 300:
    #                     sz = 1
    #                 elif props[labels[y0, x0] - 1].area < 2000:
    #                     sz = 2
    #             uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
    #                              max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
    #             if len(uniq[uniq > 0]) > 1:
    #                 msk1[y0, x0] = True
    #                 msk0[y0, x0] = 0
    #
    #     msk1 = 255 * msk1
    #     msk1 = msk1.astype('uint8')
    #
    #     msk2 = np.zeros_like(labels, dtype='uint8')
    #     msk = np.stack((msk0, msk1, msk2))
    #     msk = np.rollaxis(msk, 0, 3)
    #     return msk