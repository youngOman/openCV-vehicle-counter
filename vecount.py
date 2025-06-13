# python vecount.py -p mobilenet_ssd/MobileNetSSD_deploy.prototxt -m mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/traffic.mp4
# -p mobilenet_ssd/MobileNetSSD_deploy.prototxt -m mobilenet_ssd/MobileNetSSD_deploy.caffemodel
from pyimagesearch.centroidtracker import CentroidTracker  # 追蹤物體軌跡 判斷位置
from pyimagesearch.trackableobject import TrackableObject
import paho.mqtt.client as mqtt  # mqtt連線
import json  # mqtt傳送訊息的格式
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import numpy as np
import argparse
import pymysql
import datetime
import time
import dlib
import imutils
# 指名讓我們的程式接受哪些命令列參數
client = mqtt.Client()
client.connect("127.0.0.1", 1883, 60)

ap = argparse.ArgumentParser(description="輸入影片檔")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])  # DNN模組載入caffe訓練好的SSD模型 引數對應定義模型結構的prototxt檔案，第二個引數對應於訓練好的model


min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 550
matches = []
cars = 0


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy
    # return (cx, cy)


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(args["input"])  # 若要指定影片參數改('car.mp4')
# cap = cv2.VideoCapture('traffic.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

if not args.get("input", False):
    print("[INFO] starting video stream...")
    cap = VideoStream(src=0).start()
    time.sleep(2.0)

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

frame = cap.read()
frame = frame[1] if args.get("input", False) else frame
frame = imutils.resize(frame, width=1000)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 物件辨識框顏色
blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
net.setInput(blob)
detections = net.forward()

if W is None or H is None:
    (H, W) = frame.shape[:2]
if totalFrames % args["skip_frames"] == 0:
    status = "Detecting"
    trackers = []
    # cv2.dnn.blobFromImage image:輸入圖像（1、3或者4通道）
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)  # 載入完之後，使用blobFromImage函式，將圖片轉換成blob格式
    net.setInput(blob)
    detections = net.forward()  # 通過forward()函式進行前向傳播
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:  # 相似值低於一定數值及不辨識
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "car":  # 非車不辨識
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])  # 計算辨識框位置
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)  # 相似度
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            trackers.append(tracker)
if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()
while ret:
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(grey,(5,5),0)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    # ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue
        cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)  # 長方形辨識框大小
        cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)  # 臨界線
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cx, cy = get_centroid(x, y, w, h)
        for (x, y) in matches:
            if y > (line_height-offset) and y < (line_height+offset):
                datetime_str = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
                cars = cars+1
                matches.remove((x, y))
                totalUp += 1
                up = "up"
                db = pymysql.connect(
                    host="192.168.1.101",
                    port=3306,
                    user="www",
                    password="Buty_2350973",
                    db="vehicle_count",
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor)
                cursor = db.cursor()
                sql = "INSERT INTO `counter`(`time`,`population`) VALUES('"+datetime_str+"','"+str(totalUp)+"')"
                payload = {'time': datetime_str, 'population': str(totalUp)}
                client.publish("jack/young", json.dumps(payload))
                cursor.execute(sql)
                print(sql)
                db.commit()
                db.close()
    cv2.putText(frame1, "Total Cars Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)
    cv2.putText(frame1, "MechatronicsLAB.net", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 170, 0), 2)
    # cv2.drawContours(frame1,contours,-1,(0,0,255),2)
    cv2.imshow("Original", frame1)
    cv2.imshow("Difference", th)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    frame1 = frame2
    ret, frame2 = cap.read()
# print(matches)
cv2.destroyAllWindows()
cap.release()
