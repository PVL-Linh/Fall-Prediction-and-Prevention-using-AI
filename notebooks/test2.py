
import mediapipe as mp
import numpy as np
import threading


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils



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


def detect(model, lm_list):
    global label, confidence_scores
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    a = ''
    #labels = ["walking" ,"boxing","","" ,'nguoiTe']
    """if results[0][0] > 0.3:
        a = labels[0]
        #print(labels[0])
    elif results[0][1] >= 1:
        a = labels[1]
        #print(labels[1])
    elif results[0][2] > 1:
        a = labels[2]
        #print(labels[2])
    elif results[0][3] > 1:
        a = labels[3]
        #print(labels[3])
    elif results[0][4] > 0.7:
        a = labels[4]
        
    else:
        print("không tìm thấy !!!")

    return a"""
    labels = ["walking" ,'nguoiTe']
    if results[0][0] >0.7:
        print(labels[0], " ",confidence_scores ) 
    elif results[0][1] >0.6:
        a =(labels[1])
    else:
        print("không tìm thấy !!!")

    return a


    


