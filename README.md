# OpenCV 高速公路即時車流量計數程式

使用 OpenCV、MobileNet SSD、Caffe 深度學習模型和 dlib 追蹤技術的車輛計數系統，能夠讀取影片檔案或即時攝影畫面，自動偵測並統計通過畫面中指定區域的車輛數量

搭配 CentroidTracker 多對象質心追蹤演算法，當車輛穿越設定的計數線時自動計數，並將結果連同時間戳記儲存到 MySQL 資料庫中

# 流程

1. 影片輸入（檔案 / 攝影機）
2. 影像處理（OpenCV）
3. SSD 檢測（MobileNet）
4. 物件追蹤（dlib）
5. 計數邏輯（偵測越線 +1）
6. 資料儲存（MySQL）

# 影像處理流程

- 幀差異檢測 - 使用 `cv2.absdiff()` 比較前後影像找出有在移動的物件(`把連續兩張影像相減，看哪裡有變化`)
- 雜訊清理 - 高斯模糊 + 二值化處理
  - 使用 `cv2.GaussianBlur()` 平滑影像，減少雜訊干擾
  - 使用 `cv2.threshold()` 將影像轉為二值圖，便於後續處理
- 形態學操作 - 膨脹和閉合操作，讓偵測到的物件形狀更完整清楚
- 輪廓檢測 - 找出移動物件的邊界
- 尺寸過濾 - 只保留大小像車的物件

# features

- 影像差異檢測：比較前後影像找出有在移動的東西
- 輪廓分析：過濾掉太小的物件，只留下可能是車的
- 越線計數：設一條線，車子經過就 +1
- 即時顯示：視覺化檢測結果和計數資訊
- 資料庫儲存：記錄時間戳和計數資料，存到資料庫

# 參數調整

- min_contour_width = 40 # 最小物件寬度
- min_contour_height = 40 # 最小物件高度
- line_height = 550 # 計數線位置
- offset = 10 # 計數容差範圍


# demo

越過計數線的車輛會自動計數，並在畫面上顯示當前計數結果。程式會持續追蹤車輛，直到畫面或影片結束

![demo](https://github.com/youngOman/openCV-vehicle-counter/blob/images/count_result.png)

查看是否有成功 INSERT 計數結果到資料庫中，若有的話 TERMINAL 應會顯示計數數量與時間

![demo](https://github.com/youngOman/openCV-vehicle-counter/blob/images/cmd_insert_message.png)

去 MySQL 資料庫中也能看到計數結果，之後就能很方便地針對這些資料進行進一步的分析或資料統計+視覺化

![demo](https://github.com/youngOman/openCV-vehicle-counter/blob/images/mysql.png)

