import numpy as np
from tensorflow.keras.preprocessing.image import Iterator
import time
import os
import xml.etree.ElementTree as ET
import cv2
import pydicom as dicom


def make_mask(image, image_id, nodules):
    height, width = image.shape
    # print(image.shape)
    filled_mask = np.full((height, width), 0, np.uint8)
    contoured_mask = np.full((height, width), 0, np.uint8)
    # todo OR for all masks
    for nodule in nodules:
        for roi in nodule['roi']:
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                cv2.fillPoly(filled_mask, np.int32([np.array(edge_map)]), 255)
                # cv2.polylines(contoured_mask, np.int32([np.array(edge_map)]), color=255, isClosed=False)

    # mask = np.swapaxes(np.array([contoured_mask, filled_mask]), 0, 2)
    # cv2.imwrite('kek0.jpg', image)
    # cv2.imwrite('kek1.jpg', filled_mask)
    return np.reshape(filled_mask, (height, width, 1)) / 255


def test(a, b):
    root = '/Users/mkryuchkov/lung-ds/3000566-03192'
    nodules = parseXML('/Users/mkryuchkov/lung-ds/3000566-03192')
    image = cv2.imread('/Users/mkryuchkov/lung-ds/000001.jpg')
    for im_name in os.listdir(root):
        if not im_name.endswith('dcm'):
            continue
        image, dcm_ds = imread(root + '/' + im_name)
        print(dcm_ds.SliceLocation)
        if dcm_ds.SliceLocation == a:
            print(im_name)
            return make_mask(image, dcm_ds.SOPInstanceUID, nodules, b)
            # break
        # print(dcm_ds.get('UID'))

    # return make_mask(image, image_id, nodules)


def imread(image_path):
    ds = dicom.dcmread(image_path)
    img = ds.pixel_array
    img_2d = img.astype(float)
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)
    image = img_2d_scaled
    return image, ds


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
    xml_file = None
    for file in file_list:
        if '.' in file and file.split('.')[1] == 'xml':
            xml_file = file
            break
    prefix = "{http://www.nih.gov}"
    if xml_file is None:
        print('SCAN PATH: {}'.format(scan_path))
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
    def __init__(self, image_dir, batch_size, val_len, data_shape=(64, 64), grid_size=1, parts_number_to_include=1):
        seed = np.uint32(time.time() * 1000)
        self.image_dir = image_dir
        self.image_ids = self.create_image_ids()
        n = len(self.image_ids)
        self.val_len = val_len
        self.index_list = np.arange(n)
        np.random.shuffle(self.index_list)
        self.val_index_array = self.index_list[:val_len]
        self.index_list = self.index_list[val_len:]
        self.val_i = 0
        self.train_i = 0
        self.grid_size = 4
        self.parts_number_to_include = parts_number_to_include

        self.data_shape = data_shape
        print("total len: {}".format(n))
        print("train index array: {}".format(len(self.index_list)))
        print("val index array: {}".format(len(self.val_index_array)))
        super().__init__(n, batch_size, False, seed)

    def train_generator(self):
        def index_inc_function():
            prev = self.train_i
            self.train_i += self.batch_size // 2
            return prev, self.train_i

        return self.generator(index_inc_function, self.index_list)

    def val_generator(self):
        def index_inc_function():
            prev = self.val_i
            self.val_i += self.batch_size // 2
            return prev, self.val_i

        return self.generator(index_inc_function, self.val_index_array)

    def generator(self, index_inc_function, index_list):
        def gen():
            while 1:
                batch_x = []
                batch_y = []
                index, next_index = index_inc_function()
                index_array = index_list[index: next_index]
                for image_index in index_array:
                    file_name, parent_name = self.image_ids[image_index]
                    image, dcm_ds = imread(file_name)
                    h, w = image.shape
                    if 2022 == h or 2022 == w:
                        hpad = (2048 - h) // 2
                        wpad = (2048 - w) // 2
                        image = np.pad(image, ((hpad, hpad),  (wpad, wpad)), constant_values=0)
                    nodules = parseXML(parent_name)
                    mask = make_mask(image, dcm_ds.SOPInstanceUID, nodules)
                    image_parts, mask_parts = self.split(image, mask)
                    for i in range(2):
                        image = image_parts[i]
                        mask = mask_parts[i]
                        cv2.imwrite(dcm_ds.SOPInstanceUID + '.png', mask)
                        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
                        image = np.repeat(image, 3, axis=2)
                        image = cv2.resize(image, self.data_shape)
                        mask = cv2.resize(mask, self.data_shape)
                        mask = np.reshape(mask, (self.data_shape[0], self.data_shape[1], 1))
                        batch_x.append(image)
                        batch_y.append(mask)
                batch_x = np.array(batch_x, dtype=np.uint8)
                batch_y = np.array(batch_y, dtype=np.uint8)
                yield batch_x, batch_y

        return gen

    def split(self, image, mask):
        h, w = image.shape
        gs = h // self.grid_size
        print(image.shape)
        image_parts = image.reshape(h // gs, gs, -1, gs).swapaxes(1, 2).reshape(-1, gs, gs)
        mask_parts = mask.reshape(h // gs, gs, -1, gs).swapaxes(1, 2).reshape(-1, gs, gs)
        max_part_idx = np.argmax([part.max() for part in mask_parts])
        rand_idx = np.random.randint(16)
        max_mask = mask_parts[max_part_idx]

        print('non_zero values in mask: {}'.format(np.count_nonzero(max_mask > 0) / max_mask.size))

        # print(max_part_idx)

        return [image_parts[max_part_idx], image_parts[rand_idx]], [max_mask,
                                                                    mask_parts[
                                                                        rand_idx]]

    def create_image_ids(self):
        dcms = []
        for root, folders, files in os.walk(self.image_dir):
            has_xml = False
            for file in files:
                if 'xml' in file:
                    has_xml = True
                    break
            if not has_xml:
                continue
            for file in files:
                if file.endswith('dcm'):
                    dcms.append((root + '/' + file, root))
        image_ids = {}
        print('total training ds len: {}'.format(len(dcms)))
        for i, dcm in enumerate(dcms):
            image_ids[i] = dcm
        return image_ids
