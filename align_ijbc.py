'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from absl import app, flags, logging
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import tqdm
import csv
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

root_path = '/media/Storage/facedata/ijbc/'
save_path = root_path + 'images_cropped/'
save_path_align = root_path + 'images_cropped_align/'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def load_data_from_dir(save_path, BATCH_SIZE=128, img_ext='png'):
    def transform_test_images(img):
        img = tf.image.resize(img, (112, 112))
        img = img / 255
        return img

    def get_label_withname(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        #         wh = tf.strings.split(parts[-1], ".")[0]
        wh = parts[-2]
        return wh

    def process_path_withname(file_path):
        label = get_label_withname(file_path)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = transform_test_images(img)
        return img, label

    list_gallery_ds = tf.data.Dataset.list_files(save_path + '*/*.' + img_ext, shuffle=False)
    labeled_gallery_ds = list_gallery_ds.map(lambda x: process_path_withname(x))
    dataset = labeled_gallery_ds.batch(BATCH_SIZE)
    return dataset

def main(_):
    dataset = load_data_from_dir(save_path, BATCH_SIZE=128,
                                 img_ext='png')
    nn =0
    for image_batch, label_batch in tqdm.tqdm(dataset):
        face = detector.detect_faces(image_batch)
        for i in range(face.shape[0]):
            nn = nn + 1
            face_i = face[i]
            img1 = image_batch[i]
            my_save_path = save_path_align + str(label_batch[i]) + '/' + str(nn) + '.png'
            create_dir(my_save_path)
            try:
                # bounding box
                bounding_box = face_i["box"]
                # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
                face1 = img1[int(bounding_box[1]):int(bounding_box[1]) + int(bounding_box[3]),
                        int(bounding_box[0]):int(bounding_box[0]) + int(bounding_box[2])]
                face1 = cv2.resize(face1, (112, 112))
                cv2.imwrite(my_save_path, face1)

            except:
                logging.info(f'fail !! {img1}')
                face1 = cv2.resize(img1, (112, 112))
                cv2.imwrite(my_save_path, face1)

if __name__ == '__main__':
    app.run(main)
