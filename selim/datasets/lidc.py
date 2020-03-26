import numpy as np
from tensorflow.keras.preprocessing.image import Iterator
import time
import os
import xml.etree.ElementTree as ET
import cv2
import pydicom as dicom


def make_mask(image, image_id, nodules):
    height, width, depth = image.shape
    nodule_image = np.zeros((height, width), np.uint8)
    # todo OR for all masks
    for nodule in nodules:
        print(nodule)
        for roi in nodule['roi']:
            print(roi)
            if roi['sop_uid'] == image_id:
                edgeMap = roi['xy']

    cv2.fillConvexPoly(nodule_image, np.array(edgeMap), (122, 122, 122))
    masked_data = cv2.bitwise_and(image, image, mask=nodule_image)
    cv2.imwrite('1.jpg', masked_data)
    return masked_data


def test():
    roi = {'sop_uid': 1, 'xy': [[100, 500], [500, 500], [500, 100]]}
    nodules = [
        {'roi': [roi]}
    ]
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
    return img_2d_scaled


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
    for file in file_list:
        if file.split('.')[1] == 'xml':
            xml_file = file
            break
    prefix = "{http://www.nih.gov/idri}"
    tree = ET.parse(scan_path + '/' + xml_file)
    root = tree.getroot()
    readingSession_list = root.findall(prefix + "CXRreadingSession")
    nodules = []

    for session in readingSession_list:
        # print(session)
        unblinded_list = session.findall(prefix + "unblindedRead")
        print(unblinded_list)
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
        self.image_name_template = image_dir + '/{}'
        self.image_dir = image_dir
        self.image_ids = self.create_image_ids()
        self.nodules = parseXML(self.image_dir)
        super().__init__(n, batch_size, False, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        for image_index in index_array:
            image_name = self.image_name_template.format(self.image_ids[image_index])
            image = imread(image_name)
            nodules = parseXML(self.image_dir)

            batch_x.append(image)
            batch_y.append(make_mask(image_name, nodules[0]['roi']))
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        return batch_x, batch_y

    def create_image_ids(self):
        dcms = filter(lambda name: name.endswith('.dcm'), os.listdir(self.image_dir))
        image_ids = {}
        for i, dcm in enumerate(dcms):
            image_ids[i] = dcm
        return image_ids