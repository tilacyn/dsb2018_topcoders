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
        print(nodule)
        for roi in nodule['roi']:
            print(roi)
            # todo ==
            if roi['sop_uid'] != image_id:
                edge_map = roi['xy']
                break

    if edge_map is None:
        return image

    cv2.fillPoly(nodule_image, np.int32([np.array(edge_map)]), (122, 122, 122))
    masked_data = cv2.bitwise_and(image, image, mask=nodule_image)
    # cv2.imwrite('1.jpg', masked_data)
    print("\n\nmask created\n\n")
    return masked_data


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
    image3d = np.zeros([image.shape[0], image.shape[1], 3])
    for i in range(len(image)):
        for j in range(len(image[i])):
            image3d[i][j] = [image[i][j], image[i][j], image[i][j]]

    return np.array(image3d), ds

def kek():

    for root, subf, files in os.walk('..'):
        print(root, subf, files)


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
        super().__init__(n, batch_size, False, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        for image_index in index_array:
            file_name, parent_name = self.image_ids[image_index]
            image, dcm_ds = imread(file_name)
            nodules = parseXML(parent_name)
            print('processing image: {}'.format(file_name))
            batch_x.append(image)
            batch_y.append(make_mask(image, dcm_ds.get('UID'), nodules))
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        print("batch_x.shape:")
        print(batch_x.shape)
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