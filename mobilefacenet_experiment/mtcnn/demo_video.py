import cv2
import os
import time
from PIL import Image
from mtcnn.detector import detect_faces
from mtcnn.models import PNet, RNet, ONet
import torch
os.chdir('..')
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of numpy image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    """
    for b in bounding_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

    return img

# LOAD MODELS
pnet = PNet().to(device)
rnet = RNet().to(device)
onet = ONet().to(device)

if __name__ == '__main__':
    video_src = './mtcnn/video/1.mp4' # video source
    # video_src = 0 # camera device id
    capture = cv2.VideoCapture(video_src)
    if not capture.isOpened():
        print('Camera is not opened!')
    else:
        idx_frame = 0
        while True:
            ret, frame = capture.read()
            idx_frame += 1
            if idx_frame % 2 != 0:
                continue
            idx_frame =0
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start = time.time()
            img = Image.fromarray(frame)
            try:
                bounding_boxes, landmarks = detect_faces(img, pnet=pnet, rnet=rnet, onet=onet, device=device)
            except:
                continue
            end = time.time()
            print(end - start)
            frame = show_bboxes(frame, bounding_boxes, landmarks)
            cv2.imshow('Video', frame)
            cv2.waitKey(1)

    capture.release()
