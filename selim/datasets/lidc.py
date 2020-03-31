import numpy as np
from tensorflow.keras.preprocessing.image import Iterator
import time
import os
import xml.etree.ElementTree as ET
import cv2
import pydicom as dicom

from matplotlib import pyplot as plt

from functools import reduce

def make_mask(image, image_id, nodules):
    height, width = image.shape
    # print(image.shape)
    filled_mask = np.full((height, width), 0, np.uint8)
    contoured_mask = np.full((height, width), 0, np.uint8)
    # todo OR for all masks
    edge_map = None
    for nodule in nodules:
        for roi in nodule['roi']:
            if roi['sop_uid'] == image_id:
                edge_map = roi['xy']
                # print(edge_map)
                cv2.fillPoly(filled_mask, np.int32([np.array(edge_map)]), 255)
                cv2.polylines(contoured_mask, np.int32([np.array(edge_map)]), color=255, isClosed=False)

    mask = np.swapaxes(np.array([contoured_mask, filled_mask]), 0, 2)
    # cv2.imwrite('kek0.jpg', mask[:,:,0])
    # cv2.imwrite('kek1.jpg', mask[:,:,1])
    return mask / 255

def test(a):
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
            return make_mask(image, dcm_ds.SOPInstanceUID, nodules)
            # break
        # print(dcm_ds.get('UID'))

    # return make_mask(image, image_id, nodules)


def imread(image_path):
    ds = dicom.dcmread(image_path)
    img = ds.pixel_array
    img_2d = img.astype(float)

    ## Step 2. Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    ## Step 3. Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)
    image = img_2d_scaled
    # cv2.imwrite('image3d.jpg', image)
    # image3d = cv2.imread('image3d.jpg')
    # print(image.shape)
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
        self.image_dir = image_dir
        self.image_ids = self.create_image_ids()
        n = len(self.image_ids)
        self.data_shape = (256, 256)
        print("total len: {}".format(n))
        super().__init__(n, batch_size, False, seed)

    def generator(self):
        def gen():
            while 1:
                batch_x = []
                batch_y = []
                index_array = np.random.choice(self.n, self.batch_size, replace=False)
                for image_index in index_array:
                    file_name, parent_name = self.image_ids[image_index]
                    image, dcm_ds = imread(file_name)
                    nodules = parseXML(parent_name)
                    # print('processing image: {}'.format(file_name))
                    mask = make_mask(image, dcm_ds.SOPInstanceUID, nodules)
                    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
                    image = np.repeat(image, 3, axis=2)
                    image = cv2.resize(image, self.data_shape)
                    mask = cv2.resize(mask, self.data_shape)
                    batch_x.append(image)
                    batch_y.append(mask)
                batch_x = np.array(batch_x, dtype=np.uint8)
                batch_y = np.array(batch_y, dtype=np.uint8)
                yield batch_x, batch_y
        return gen

    def create_image_ids(self):
        dcms = []
        observed = self.list_observed()
        # files_for_training = ['000041.dcm', '000071.dcm', '000104.dcm', '00003.dcm']
        files_for_training = ['000071.dcm']
        for root, folders, files in os.walk(self.image_dir):
            if not '3000566-03192' in root:
                continue
            # if not reduce(lambda x, y: x or y, [dir_substr in root for dir_substr in observed]):
            #     continue
            for file in files:
                # if file.endswith('dcm'):
                if file in files_for_training:
                    dcms.append((root + '/' + file, root))
        image_ids = {}
        print('total training ds len: {}'.format(len(dcms)))
        for i, dcm in enumerate(dcms):
            image_ids[i] = dcm
        return image_ids

    def list_observed(self):
        return ['0787', '0356', '0351', '0292', '0287', '0272', '0560', '0509', '0502', '0499',  '0480', '0404']


class LIDCValidationDatasetIterator(LIDCDatasetIterator):
    def list_observed(self):
        return ['0941', '0848', '0796']


