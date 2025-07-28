from ultralytics import YOLO
import cv2
#import cvzone
import math
from src.data_processing.sort import Sort
import numpy as np
import winsound



# tải mô hình YOLO
model = YOLO("model/yolov8n.pt")

# Đọc nội dung từ tệp a.txt  xác định tên lớp để phát hiện đối tượng
with open('className.txt', 'r') as file:
    classNames = file.read().splitlines()

# khởi tạo tracker sort để theo dõi các đối tượng
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# xác định khu vực giới hạn
top = [0, 160, 250, 160]
bottom = [0, 450, 250, 450]

def in_out(img):
   


    # sử dụng mặt nạ - chỉ tập trung vào vùng cần quan sát
    tempImg = img[150:150 + 700, 130:130 + 250]
    # Đây là một đối tượng mô hình đã được tạo từ thư viện Ultralytics YOLO
    results = model(tempImg, stream=True)

    # mảng lưu trữ kết quả nhận diện
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Trích xuât tọa độ bouding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # trích xuất độ tin cậy & vị trí của tên lớp
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # kiểm tra nếu tên lớp là person & độ tin cậy thõa mãn
            if currentClass == "person" and conf > 0.7:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # trả về số lượng & cập nhật bouding box trong khung hình hiện tại
    resultsTracker = tracker.update(detections)


    # đi qua đường thẳng này sẽ được đếm
    cv2.line(tempImg, (top[0], top[1]), (top[2], top[3]), (0, 0, 255), 5)
    cv2.line(tempImg, (bottom[0], bottom[1]), (bottom[2], bottom[3]), (0, 0, 255), 5)


    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1


    

        # vẽ điểm trung tâm của bounding box
        cx, cy = x1 + w // 2, y1
        cx1, cy1 = x1 + w // 2, y1 + h
  
        label = ''
        if bottom[0] < cx1 < bottom[2] and cy1 < bottom[1] - 10:
            if cy > top[1]:
                winsound.Beep(10000, 500)  # (Tần số 1000 Hz, thời gian 500 ms)
                label= "Children"
            else:
                label= "Adult"
        return label


