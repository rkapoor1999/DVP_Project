import os
import sys
import cv2
import numpy as np
from collections import Counter

from detect_faces_dlib import detect_faces, detect_faces_tolist
from character_detection import textInImage
# import easyocr

# Initialize Kalman filters and tracking points
kalman_filters = {}  # Dictionary to store Kalman filter instances with unique identifiers
det_num_list = []
pt = {}  # Dictionary to store tracking points with unique identifiers
next_object_id = 0   # Counter to assign unique identifiers to new objects


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out=None

#Here I use my second camera, you can simply give a video address instead
cap = cv2.VideoCapture('drdavid.mp4') 
#cap = cv2.VideoCapture('apple.mp4') 
# cap = cv2.VideoCapture('us.MOV')

frameCounter = 0
num_frames = 10

# ret ,frame = cap.read()
folder_name = 'david'

# Check if the folder exists
if not os.path.exists(folder_name):
    # If it doesn't exist, create it
    os.makedirs(folder_name)


num_face=0
while True:
    ret ,frame = cap.read()

    if not ret:
        break
    filename = f'{folder_name}/david_{frameCounter}_face_char_kalman_multi.png'
    # filename = f'res_mult_test/testus_frame_{frameCounter}_face_kalman_multi3.png'
    print("FRAME NUM:", frameCounter)
    detected_faces = detect_faces_tolist(frame)
    sorted_faces = sorted(detected_faces, key=lambda box: box[0])

    for i, face in enumerate(sorted_faces):
        # object_id = f'face_{next_object_id}'
        object_id = f'face_{i}'
        print("face and object id", face, object_id)
        if object_id not in kalman_filters:
            print(len(kalman_filters), len(detect_faces(frame)))
            print("Adding new object id:", object_id)
            # c,r,w,h = face.left(), face.top(), face.width(), face.height()
            c,r,w,h = face[0],face[1],face[2],face[3]  
            print("track point of the first frame", c,r,w,h)
            #initialize the KF
            kalman_filters[object_id] = cv2.KalmanFilter(4,2,0)   # 4 - dimensionality of state, 2 - dimensionality of measurement
            kalman_filters[object_id].transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                                                [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                                                [0., 0., 1., 0.],
                                                                [0., 0., 0., 1.]])
            kalman_filters[object_id].measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
            kalman_filters[object_id].processNoiseCov = 1e-6 * np.eye(4, 4)      # respond faster to change and be less smooth
            kalman_filters[object_id].measurementNoiseCov = 1e-2 * np.eye(2, 2)
            kalman_filters[object_id].errorCovPost = 1e-1 * np.eye(4, 4)
            kalman_filters[object_id].statePost = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position

            pt[object_id] = (frameCounter, c+w/2, r+h/2)
            print("track point of the first frame", pt[object_id])
            
            prediction = kalman_filters[object_id].predict()
            print("prediction:", prediction)
            # next_object_id +=1

        print("number of detected objects:", len(sorted_faces))
        det_num_list.append(len(sorted_faces))

    counts = Counter(det_num_list)
    most_frequent_num = max(counts, key=counts.get)
    print("most frequent:", most_frequent_num)
    # there are some faces not detected 
    if len(sorted_faces)>=most_frequent_num:
        print("CASE1: detected all!!!!")
        for i, face in enumerate(sorted_faces):
            object_id = f'face_{i}'
            c,r,w,h = face  
            #draw red boxes
            img2 = cv2.circle(frame, (c+w//2, r+h//2), 2, (0,0,255), -1)
            img2=cv2.rectangle(frame,(c,r),(c+w,r+h),(0,0,255),2) 
            # apply Kalman to adjust
            measurement = np.matrix(np.array([c+w/2, r+h/2], dtype='float64')).transpose()
            posterior = kalman_filters[object_id].correct(measurement)
            pt[object_id] = (frameCounter, int(posterior[0]), int(posterior[1]))
            print("pt of {}:".format(object_id),  pt[object_id])
            #draw green boxes
            img2 = cv2.circle(frame, (pt[object_id][1], pt[object_id][2]), 2, (0,255,0), -1)
            img2=cv2.rectangle(frame,(pt[object_id][1]-w//2,pt[object_id][2]-h//2),(pt[object_id][1]+w//2,pt[object_id][2]+h//2),(0,255,0),2)
    else:
        print("CASE2: there are missing faces. Use Kalman:")
        for i in range(most_frequent_num):
            object_id = f'face_{i}'
            prediction = kalman_filters[object_id].predict()
            pt[object_id] = (frameCounter, int(prediction[0]), int(prediction[1]))
            print("pt of {}:".format(object_id),  pt[object_id])
            #draw green boxes
            img2 = cv2.circle(frame, (pt[object_id][1], pt[object_id][2]), 2, (0,255,0), -1)
            img2=cv2.rectangle(frame,(pt[object_id][1]-w//2,pt[object_id][2]-h//2),(pt[object_id][1]+w//2,pt[object_id][2]+h//2),(0,255,0),2)

    detected_texts = textInImage(frame)
    sorted_texts = sorted(detected_texts, key=lambda box: box[0])
    # reader = easyocr.Reader(["en"],gpu=False)
    print("###Detected texts: ", sorted_texts)
    for i, text in enumerate(sorted_texts):
        c2,r2,w2,h2 = text
        #draw red boxes
        img2 = cv2.circle(frame, (c2+w2//2, r2+h2//2), 2, (255,0,0), -1)
        img2=cv2.rectangle(frame,(c2,r2),(c2+w2,r2+h2),(255,0,0),2) 



    # save as image
    if 'img2' in locals():
        cv2.imwrite(filename, img2)
    else:
        cv2.imwrite(filename, frame)
    frameCounter += 1
    if frameCounter == num_frames:
        break

    
    

 
