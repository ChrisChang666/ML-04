import time
import sys
import pathlib
import logging
import cv2
from tools.test_detect import MtcnnDetector
import torch


def draw_images(img, bboxs, landmarks):  # 在图片上绘制人脸框及特征点
    num_face = bboxs.shape[0]
    for i in range(num_face):
        cv2.rectangle(img, (int(bboxs[i, 0]), int(bboxs[i, 1])), (int(bboxs[i, 2]), int(bboxs[i, 3])), (0, 255, 0), 2)
    for p in landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[2 * i]), int(p[2 * i + 1])), 2, (0, 0, 255), -1)
    return img

def CatchUsbVideo(window_name, video_src, mtcnn_detector):
    cv2.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    # cap = cv2.VideoCapture(0)  # 视频地址
    cap = cv2.VideoCapture(video_src)
    
    frameNum = 0
    with torch.no_grad():
        while cap.isOpened():
            ok, img = cap.read() #读取一帧数据
            frameNum += 1
            if frameNum % 8 !=0:
                continue
            frameNum = 0
            if not ok:
                break


            RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t1 = time.time()
            bboxs, landmarks = mtcnn_detector.detect_face(RGB_image)  # 检测得到bboxs以及特征点
            img = draw_images(img, bboxs, landmarks)  # 得到绘制人脸框及特征点的图片
            t2 = time.time()
            print(' timer:{}'.format( t2 - t1))

            #显示图像
            cv2.imshow(window_name, img)
            c = cv2.waitKey(8)
            if c & 0xFF == ord('q'):
                break

        #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False)  # 加载模型参数，构造检测器,如果有gpu，设置use_cuda=True, 否则use_cuda=False


    # 用视频或者摄像头获取图片
    video_src = './data/test_video/DonaldTrump.mp4' # 视频文件的路径，检测视频中的人脸
    # video_src = 0 # 摄像头的id,使用摄像头实时检测

    if len(sys.argv) != 1:
            print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("face_window",video_src, mtcnn_detector)