# python vecount.py -p mobilenet_ssd/MobileNetSSD_deploy.prototxt -m mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/traffic.mp4
from pyimagesearch.centroidtracker import CentroidTracker  # 追蹤物體軌跡 判斷位置
from pyimagesearch.trackableobject import TrackableObject
# import json  # mqtt傳送訊息的格式
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

import os
# pip install python-dotenv
from dotenv import load_dotenv

# 載入環境變數(.env檔)
load_dotenv()

# 設定命令列參數解析器
ap = argparse.ArgumentParser(description="輸入影片檔")
ap.add_argument("-i", "--input", type=str,
                help="選擇性輸入影片檔案的路徑")
ap.add_argument("-p", "--prototxt", required=True,
                help="Caffe 'deploy' prototxt 檔案路徑")
ap.add_argument("-m", "--model", required=True,
                help="Caffe 預訓練模型路徑")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="過濾弱檢測的最小機率閾值")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="檢測之間跳過的幀數")
args = vars(ap.parse_args())

# 定義物件類別
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# 為每個類別生成隨機顏色
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# 載入 Caffe 深度神經網路模型
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])  # DNN模組載入caffe訓練好的SSD模型 引數對應定義模型結構的prototxt檔案，第二個引數對應於訓練好的model

# 設定最小輪廓尺寸參數
min_contour_width = 40
min_contour_height = 40
offset = 10  # 計數線的容差範圍
line_height = 550  # 計數線的高度位置
matches = []  # 儲存匹配的質心點
cars = 0  # 車輛計數器

# 計算邊界框的質心座標


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# 初始化視訊捕獲
cap = cv2.VideoCapture(args["input"])  # 若要指定影片參數改('car.mp4')
# 設定視訊解析度
cap.set(3, 1920)
cap.set(4, 1080)

# 如果沒有輸入檔案，使用攝影機
if not args.get("input", False):
    print("[INFO] 啟動視訊串流...")
    cap = VideoStream(src=0).start()
    time.sleep(2.0)

# 初始化影像尺寸變數
W = None
H = None

# 初始化質心追蹤器
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []  # 追蹤器列表
trackableObjects = {}  # 可追蹤物件字典

# 初始化計數器
totalFrames = 0
totalDown = 0
totalUp = 0

# 讀取第一幀用於初始化
frame = cap.read()
frame = frame[1] if args.get("input", False) else frame
frame = imutils.resize(frame, width=1000)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 物件辨識框顏色
blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
net.setInput(blob)
detections = net.forward()

# 設定影像尺寸
if W is None or H is None:
    (H, W) = frame.shape[:2]

# 執行物件檢測
if totalFrames % args["skip_frames"] == 0:
    status = "Detecting"  # 檢測狀態
    trackers = []
    # 將影像轉換為 blob 格式進行神經網路處理
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)  # 載入完之後，使用blobFromImage函式，將圖片轉換成blob格式
    net.setInput(blob)
    detections = net.forward()  # 通過forward()函式進行前向傳播

    # 遍歷所有檢測結果
    for i in np.arange(0, detections.shape[2]):
        # 提取與預測相關的信心度（機率）
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:  # 相似值低於一定數值及不辨識
            # 從檢測列表中提取類別標籤的索引
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "car":  # 非車不辨識
                continue
            # 計算邊界框座標
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])  # 計算辨識框位置
            (startX, startY, endX, endY) = box.astype("int")
            # 在影像上繪製預測結果
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)  # 相似度
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # 從邊界框座標構造 dlib 矩形物件，然後啟動 dlib 相關追蹤器
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            # 將追蹤器加入追蹤器列表，以便在跳過幀時使用
            trackers.append(tracker)

# 檢查視訊是否成功開啟
if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False

# 讀取前兩幀用於差異檢測
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# 主要處理迴圈
while ret:
    # 計算兩幀之間的絕對差異
    d = cv2.absdiff(frame1, frame2)
    # 轉換為灰階影像
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # 應用高斯模糊以減少雜訊
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    # 二值化處理
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # 膨脹操作以填補小洞
    dilated = cv2.dilate(th, np.ones((3, 3)))
    # 建立橢圓形結構元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # 閉合操作以填補小洞
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    # 尋找輪廓
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 處理每個輪廓
    for (i, c) in enumerate(contours):
        # 取得邊界矩形
        (x, y, w, h) = cv2.boundingRect(c)
        # 檢查輪廓是否符合最小尺寸要求
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue
        # 繪製檢測框
        cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)  # 長方形辨識框大小
        # 繪製計數線
        cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)  # 臨界線
        # 計算質心
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        # 繪製質心點
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cx, cy = get_centroid(x, y, w, h)

        # 檢查物件是否穿越計數線
        for (x, y) in matches:
            if y > (line_height-offset) and y < (line_height+offset):
                # 取得當前時間
                datetime_str = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
                cars = cars+1
                matches.remove((x, y))
                totalUp += 1
                up = "up"

                # 連線資料庫並儲存計數結果
                try:
                    db = pymysql.connect(
                        host=os.getenv("DB_HOST"),
                        port=int(os.getenv('DB_PORT', 3306)),
                        user=os.getenv("DB_USER"),
                        password=os.getenv("DB_PASSWORD"),
                        db=os.getenv("DB_NAME"),
                        charset="os.getenv('DB_CHARSET', 'utf8mb4')",
                        cursorclass=pymysql.cursors.DictCursor)
                    cursor = db.cursor()
                    sql = "INSERT INTO `counter`(`time`,`population`) VALUES(%s, %s)"
                    cursor.execute(sql, (datetime_str, str(totalUp)))
                    print(f"資料已儲存: {datetime_str}, 總計: {totalUp}")
                    db.commit()
                except Exception as e:
                    print(f"資料庫連線錯誤: {e}")
                finally:
                    if db in locals():
                        db.close()

    # 在影像上顯示計數資訊
    cv2.putText(frame1, "Total Cars Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)
    cv2.putText(frame1, "MechatronicsLAB.net", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 170, 0), 2)

    # 顯示處理結果
    cv2.imshow("Original", frame1)  # 顯示原始影像
    cv2.imshow("Difference", th)    # 顯示差異影像

    # 檢查是否按下 'q' 鍵退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # 更新幀
    frame1 = frame2
    ret, frame2 = cap.read()

# 清理資源
cv2.destroyAllWindows()
cap.release()
