import cv2
import numpy as np
from collections import Counter

# Initialize Kalman filters and tracking points
kalman_filters = {}  # Dictionary to store Kalman filter instances with unique identifiers
det_num_list = []
pt = {}  # Dictionary to store tracking points with unique identifiers
next_object_id = 0   # Counter to assign unique identifiers to new objects
translation_matrix = np.array([[1., 0., .1, 0.], [0., 1., 0., .1], [0., 0., 1., 0.], [0., 0., 0., 1.]])

def initialize_kalman_filter(c,r,w,h):
    kalman_filter = cv2.KalmanFilter(4,2,0)   # 4 - dimensionality of state, 2 - dimensionality of measurement
    kalman_filter.transitionMatrix = translation_matrix
    kalman_filter.measurementMatrix = 1. * np.eye(2, 4)      # you can tweak these to make the tracker
    kalman_filter.processNoiseCov = 1e-6 * np.eye(4, 4)      # respond faster to change and be less smooth
    kalman_filter.measurementNoiseCov = 1e-2 * np.eye(2, 2)
    kalman_filter.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman_filter.statePost = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman_filter.predict()
    return kalman_filter


def kalman_filter(coordinates, frame_count):
    coordinates = sorted(coordinates, key=lambda box: box[0])
    for object_id, face in enumerate(coordinates):
        if object_id in kalman_filters:
            continue
        c,r,w,h = face[0],face[1],face[2],face[3]  
        kalman_filters[object_id] = initialize_kalman_filter(c,r,w,h)
        pt[object_id] = (frame_count, c+w/2, r+h/2, (w,h))
        
    det_num_list.append(len(coordinates))

    counts = Counter(det_num_list)
    most_frequent_num = max(counts, key=counts.get)
    
    adjusted_coordinates = []
    if len(coordinates)>=most_frequent_num:
        # print("CASE1: detected all!!!!")
        for object_id, face in enumerate(coordinates):
            c,r,w,h = face  
            measurement = np.matrix(np.array([c+w/2, r+h/2], dtype='float64')).transpose()
            posterior = kalman_filters[object_id].correct(measurement)
            pt[object_id] = (frame_count, int(posterior[0]), int(posterior[1]), pt[object_id][3])
            # adjusted_coordinates.append([pt[object_id][1]-w//2, pt[object_id][2]-h//2, pt[object_id][1]+w//2,pt[object_id][2]+h//2])
            adjusted_coordinates.append([c, r, c+w, r+h])
    else:
        # print("CASE2: there are missing faces. Use Kalman:")
        for object_id in range(most_frequent_num):
            w,h = pt[object_id][3]  
            prediction = kalman_filters[object_id].predict()
            pt[object_id] = (frame_count, int(prediction[0]), int(prediction[1]), pt[object_id][3])
            adjusted_coordinates.append([pt[object_id][1]-w//2, pt[object_id][2]-h//2, pt[object_id][1]+w//2,pt[object_id][2]+h//2])
    return adjusted_coordinates
