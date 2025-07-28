import cv2
import mediapipe as mp
import numpy as np
import threading
from keras.models import load_model

label = "Warmup...."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = load_model("model/model_te_di.h5")
video_path = "data/nguoite.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

lock = threading.Lock()
confidence_scores = []

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

def draw_class_on_image(label, confidence_scores, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    
    y0, dy = 30, 30
    for i, (lbl, score) in enumerate(confidence_scores):
        text = f"{lbl}: {score:.2f}"
        y = y0 + i * dy
        cv2.putText(img, text, (10, y), font, fontScale, fontColor, thickness, lineType)
    
    return img

def detect(model, lm_list):
    global label, confidence_scores
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)

    #labels = ["walking" ,"boxing","sit","run" ,'nguoiTe']
    labels = ["walking" ,'nguoiTe' ]
    """
    if results[0][0] >0.7:
        print(labels[0])
    elif results[0][1] >0.7:
        print(labels[1])
    elif results[0][2] >0.7:
        print(labels[2])
    elif results[0][3] >0.7:
        print(labels[3])
    #elif results[0][4] > 0.7:
        #print(labels[4])
    else:
        print("không tìm thấy !!!")
    
    """
    if results[0][0] >0.7:
        print(labels[0])
    elif results[0][1] >0.8:
        print(labels[1])
    else:
        print("không tìm thấy !!!")


    # hiển thị label màn hinh
    if results.shape[0] > 0 and results.shape[1] >= 2:
        
        confidence_scores = [(labels[i], results[0][i]) for i in range(len(labels))]
        max_index = np.argmax(results[0])

        print(confidence_scores)


        with lock:
            label = labels[max_index]
            
    
    return label

i = 0
warmup_frames = 60

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (800, 600))
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i += 1
    if i > warmup_frames:
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    with lock:
        img = draw_class_on_image(label, confidence_scores, img)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
