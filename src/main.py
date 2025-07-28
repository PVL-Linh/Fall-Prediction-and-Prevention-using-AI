from test2 import make_landmark_timestep, detect
from keras.models import load_model
from sent_mail import send_mail
from in_out_door import in_out
from ultralytics import YOLO
from tkinter import ttk
import mediapipe as mp
import tkinter as tk
import threading
import pyaudio
import wave
import time
import cv2
import os
from PIL import Image, ImageTk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def play_notification_sound():
    global notification_playing, last_notification_time
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    p = pyaudio.PyAudio()
    wf = wave.open("baodong1.wav", 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    while data:
        if not notification_playing:
            break
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()
    last_notification_time = time.time()

def stop_notification_sound():
    global notification_playing
    notification_playing = False

#=================================================================
# Khởi tạo các biến và mô hình
label = "Warmup...."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model_h5 = load_model("model/model_te_di.h5")
model_yolo = YOLO('model/yolov8n.pt')

 
#cap = cv2.VideoCapture(0)
video_path = "data/p.mp4"
cap = cv2.VideoCapture(video_path)


i = 0
warmup_frames = 60

# Biến để quản lý thời gian gửi email
last_email_time = 0
email_interval = 2 * 60  # 2 phút
# Biến toàn cục cho âm thanh cảnh báo
notification_playing = False
last_notification_time = 0
#end setup============================================================

true_button_active = False  # Cờ để xác định trạng thái của nút True

# Hàm xử lý từng khung hình
def process_frame():
    global i, lm_list, notification_playing, last_notification_time, last_email_time, true_button_active

    success, img = cap.read()
    if not success:
        return

    # Kiểm tra cờ để xác định hành động của nút nhấn
    if true_button_active:
        label = in_out(img)
        print('=======> ', label)


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_yolo = model_yolo(imgRGB)

    num_people = 0
    i += 1
    if i > warmup_frames:
        for result in results_yolo:
            boxes = result.boxes.xyxy
            scores = result.boxes.conf

            for box, score in zip(boxes, scores):
                if score > 0.3:
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    person_roi = img[y1:y2, x1:x2]
                    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    results_pose = pose.process(person_rgb)

                    if results_pose.pose_landmarks:
                        num_people += 1
                        c_lm = make_landmark_timestep(results_pose)
                        lm_list.append(c_lm)
                        a = detect(model_h5, lm_list)
                        print('============>  ', type(a))
                        
                        current_time = time.strftime("%H:%M")
                        
                        if a =="nguoiTe" or a == "boxing" :
                            result_with_time = f"      {current_time}:  {a}"
                            result_list.insert(0, result_with_time)  
                            if len(result_list.get(0, tk.END)) > 20:  
                                result_list.delete(tk.END)

                        if true_button_active and label != '' and label != None:
                            result_with_time1 = f"      {current_time}: {label}"
                            result_list.insert(0, result_with_time1)  
                            if len(result_list.get(0, tk.END)) > 20:  
                                result_list.delete(tk.END)
                    
                        # Logic cho gửi email báo động và ngừng báo động
                        if a == "boxing" or a == "nguoiTe" :
                            if not notification_playing:
                                notification_playing = True
                                threading.Thread(target=play_notification_sound).start()

                            # Nếu nhãn là "boxing" hoặc "kid", gửi email báo động mỗi 2 phút
                            if time.time() - last_email_time > email_interval:
                                threading.Thread(target=send_mail).start()
                                last_email_time = time.time()


                        elif true_button_active:
                            if label == "Children":
                                if not notification_playing:
                                    notification_playing = True
                                    threading.Thread(target=play_notification_sound).start()

                            # Nếu nhãn là "boxing" hoặc "kid", gửi email báo động mỗi 2 phút
                                if time.time() - last_email_time > email_interval:
                                    threading.Thread(target=send_mail).start()
                                    last_email_time = time.time()
                        else:

                            stop_notification_sound()

                        # Reset danh sách landmarks sau khi gửi đi
                        if len(lm_list) == n_time_steps:
                            lm_list = []
    
    

    # Resize ảnh trước khi hiển thị
    img = cv2.resize(img, (600, 500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    canvas.itemconfig(canvas_img, image=img)
    canvas.image = img

    window.after(10, process_frame)

# Hàm để kích hoạt xử lý khi nhấn nút True
def on_true_button_click():
    global true_button_active
    true_button_active = True

# Hàm để dừng xử lý khi nhấn nút False
def on_false_button_click():
    global true_button_active
    true_button_active = False

#======================================================================
window = tk.Tk()
window.title("Camera Đa chức năng")

# Tạo khung trái để hiển thị video từ camera
left_frame = ttk.Frame(window)
left_frame.grid(row=0, column=0, padx=10, pady=10)
canvas = tk.Canvas(left_frame, width=600, height=500) 
canvas.grid(row=0, column=0)
canvas_img = canvas.create_image(0, 0, anchor=tk.NW)

# Tạo khung phải để hiển thị kết quả nhãn
right_frame = ttk.Frame(window)
right_frame.grid(row=0, column=1, padx=10, pady=10)
result_list = tk.Listbox(right_frame, width=20, height=15, font=("Helvetica", 16))
result_list.grid(row=0, column=0)
#======================================================================

# Tạo khung dưới để chứa các nút True và False
button_frame = ttk.Frame(window)
button_frame.grid(row=1, column=1, columnspan=2, pady=10)
# Thêm văn bản trên các nút
false_label = ttk.Label(button_frame, text="PHÁT HIỆN VÀO CỬA")
false_label.pack(side=tk.TOP, padx=20)

# Thêm văn bản trên các nút
#true_label = ttk.Label(button_frame, text="Nhấn để chạy in_out")
#true_label.pack(side=tk.LEFT, padx=20)

# Tạo nút True
true_button = tk.Button(button_frame, text="ON", command=on_true_button_click)
true_button.pack(side=tk.LEFT, padx=20)

# Tạo nút False
false_button = tk.Button(button_frame, text="OFF", command=on_false_button_click)
false_button.pack(side=tk.RIGHT, padx=20)

# Thêm văn bản trên các nút
#true_label = ttk.Label(button_frame, text="Nhấn để tắt in_out")
#true_label.pack(side=tk.TOP, padx=20)

# Bắt đầu xử lý khung hình
window.after(10, process_frame)
window.mainloop()