import cv2
import os
import time
import numpy as np
from PIL import Image
from mtcnn.detector import detect_faces
from mtcnn.models import PNet, RNet, ONet

os.chdir('..')

import torch

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of numpy image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    """
    for b in bounding_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[i]), int(p[i + 5])), 3, (0, 255, 0), -1)
    return img


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD MODELS
pnet = PNet().to(device)
rnet = RNet().to(device)
onet = ONet().to(device)

input_shape = [112, 112]

if __name__ == '__main__':
    # dir for images to predict
    img_dir = './mtcnn/img'
    img_save = './mtcnn/img_result'
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(img_path, ' open img failed \n')
            continue
        start = time.time()
        try:
            bounding_boxes, landmarks = detect_faces(img, pnet=pnet, rnet=rnet, onet=onet ,device=device)
        except:
            print(img_path, ' no img \n')
            continue
        end = time.time()
        print(end-start)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = show_bboxes(img, bounding_boxes, landmarks)
        cv2.imwrite(os.path.join(img_save, img_name), img)


