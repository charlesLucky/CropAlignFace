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
np.random.seed(123)  # for reproducibility
from util import  *
setUpGPU()
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def to_image(arr):
    if type(arr).__module__ == 'PIL.Image':
        return arr
    if type(arr).__module__ == 'numpy':
        return Image.fromarray(arr)

def alignface(img1, ):
    # img1 = Image.fromarray(img1)
    try:
        face = detector.detect_faces(img1)
        # [{'box': [57, 71, 79, 97],
        #   'confidence': 0.9995273351669312,
        #   'keypoints': {'left_eye': (73, 110),
        #                 'right_eye': (106, 104),
        #                 'nose': (89, 133),
        #                 'mouth_left': (82, 147),
        #                 'mouth_right': (114, 141)}}]
        face = face[0]
        # bounding box
        bounding_box = face["box"]
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        face1 = img1[int(bounding_box[1]):int(bounding_box[1]) + int(bounding_box[3]),
                   int(bounding_box[0]):int(bounding_box[0]) + int(bounding_box[2])]
        face1 = cv2.resize(img1, (112, 112))
        return face1, True
    except:
        logging.info(f'fail !! {img1}')
        face1 = cv2.resize(img1, (112, 112))
        face1 = np.asarray(face1)
        return face1, False

def get_groundtruth(dataset):
    "{frame_id: [template_id, x, y, w, h]"
    frame_map = {}
    # with open(dataset, 'r', encoding='utf-8') as csvreader:
    with open(dataset, 'r') as csvreader:

        all_data = csvreader.readlines()
        for line in all_data[1:]:
            data = line.strip().split(',')
            template_id, subject_id, frame_name = data[:3]

            x, y, w, h = data[4:]
            # if 'frames' in frame_name:
            if frame_name not in frame_map:
                frame_map[frame_name] = []
            frame_data = [x, y, w, h]
            frame_map[frame_name] = frame_data

    return frame_map

def process_ijbc_frames(path_to_frames,metadata_path,save_path):

    # path_to_frames = '/media/Storage/facedata/ijbc/images/'
    # metadata_path = '/media/Storage/facedata/ijbc/protocols/ijbc_1N_probe_mixed.csv'
    # save_path = '/media/Storage/facedata/ijbc/images_cropped/'

    frames_data = get_groundtruth(metadata_path)
    nn = 0
    for frame_id, frame_data in frames_data.items():
        print(frame_id,nn)
        nn = nn +1
        x, y, w, h = frame_data
        try:
            draw = cv2.cvtColor(cv2.imread(path_to_frames + frame_id), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            continue

        y = int(y)
        x = int(x)
        w = int(w)
        h = int(h)

        face = draw[y:y + h, x:x + w]

        alignface_img,isSuccess = alignface(face)
        cv2.imwrite(save_path+frame_id, alignface_img)

    print("SUCCESS!!!!!")

def main(_):
    root_path = '/media/Storage/facedata/ijbc/'
    # root_path = '/media/charles/Storage/CropAlignFace/data/IJB-C/'
    path_to_frames = root_path + 'images/'
    metadata_path = root_path + 'protocols/ijbc_1N_probe_mixed.csv'
    save_path = root_path + 'images_cropped/'
    process_ijbc_frames(path_to_frames,metadata_path,save_path)
    metadata_path = root_path + 'protocols/ijbc_1N_gallery_G1.csv'
    process_ijbc_frames(path_to_frames,metadata_path,save_path)
    metadata_path = root_path + 'protocols/ijbc_1N_gallery_G2.csv'
    process_ijbc_frames(path_to_frames,metadata_path,save_path)


if __name__ == '__main__':
    app.run(main)
