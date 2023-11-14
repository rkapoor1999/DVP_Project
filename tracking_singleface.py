import os
import sys
import cv2
import numpy as np

from detect_faces_dlib import detect_faces


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out=None

#Here I use my second camera, you can simply give a video address instead
cap = cv2.VideoCapture('drdavid.mp4') 

frameCounter = 0
num_frames = 10

ret ,frame = cap.read()
 
# detect face in first frame
c,r,w,h = detect_faces(frame)[0].left(), detect_faces(frame)[0].top(), detect_faces(frame)[0].width(), detect_faces(frame)[0].height()
print("detected in first frame:", c,r,w,h)

#initialize the KF
kalman = cv2.KalmanFilter(4,2,0)   # 4 - dimensionality of state, 2 - dimensionality of measurement
state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                    [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)      # respond faster to change and be less smooth
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state

# Write track point for first frame
pt = (0, c+w/2, r+h/2)
print("track point for first frame", pt)


while True:
    ret ,frame = cap.read()

    if not ret:
        break
    filename = f'res/frame_{frameCounter}_face_kalman.png'

    prediction = kalman.predict()
    print("prediction:", prediction)
    
    try:
        # Attempt to detect face
        c,r,w,h = detect_faces(frame)[0].left(), detect_faces(frame)[0].top(), detect_faces(frame)[0].width(), detect_faces(frame)[0].height()
        print("detected box:", c, r, w, h)
        # original detection: only when they're available
        img2 = cv2.circle(frame, (c+w//2, r+h//2), 2, (0,0,255), -1)
        img2=cv2.rectangle(frame,(c,r),(c+w,r+h),(0,0,255),2)     
    except IndexError:
        # Handle the case where no face is detected
        print("No face detected.")

    if w != 0 and h != 0:   #measurement_valid
        measurement = np.matrix(np.array([c+w/2, r+h/2], dtype='float64')).transpose()
        print("measurement:", measurement)
        posterior = kalman.correct(measurement)
        pt = (frameCounter, int(posterior[0]), int(posterior[1]))
    else:
        # use prediction as the tracking result
        print("Nothing has been detected!")
        pt = (frameCounter, int(prediction[0]), int(prediction[1]))
    print("pt:",  pt)
    
    # corrected detection 
    img2 = cv2.circle(frame, (pt[1], pt[2]), 2, (0,255,0), -1)
    img2=cv2.rectangle(frame,(pt[1]-w//2,pt[2]-h//2),(pt[1]+w//2,pt[2]+h//2),(0,255,0),2)

    cv2.imwrite(filename, img2)
    frameCounter += 1
    if frameCounter == num_frames:
        break

    
    

 
