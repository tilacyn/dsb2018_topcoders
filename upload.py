#
# file_metadata = {'name': 'photo.jpg'}
# media = MediaFileUpload('files/photo.jpg',
#                         mimetype='image/jpeg')
# file = drive_service.files().create(body=file_metadata,
#                                     media_body=media,
#                                     fields='id').execute()
# print('File ID: %s' % file.get('id'))
import argparse
import os

import numpy as np
import pydicom as dicom
import cv2



parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--image_dir', default=None)
arg('--destination_dir', default=None)
args = parser.parse_args()

image_dir = args.image_dir
if image_dir.endswith('/'):
    image_dir = image_dir[:-1]

for file in os.listdir(args.image_dir):
    if file.endswith('dcm'):
        convert(image_dir + '/' + file)
#
# for file in os.listdir(args.image_dir):
#     if file.endswith('jpg'):
#         os.replace(image_dir + '/' + file, args.destination_dir)