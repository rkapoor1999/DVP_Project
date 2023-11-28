import os
import cv2
import sys
from detect_faces_dlib import detect_faces_tolist
from face_detection_cv2 import detect_faces
from character_detection import textInImage
from kalman_filter import kalman_filter

def read_frames(path, num_frames = -1):
    cap = cv2.VideoCapture(path)
    for x in range(num_frames):
        ret, frame = cap.read()
        if ret == True:
            yield frame
    cap.release()

def apply_blurring(image, start_x, start_y, end_x, end_y):
    if(start_x<0 or start_y<0):
        return image
    if(start_x>=end_x or start_y>=end_y):
        return image
    
    section_of_interest = image[start_y:end_y, start_x:end_x]
    blurred_section = cv2.GaussianBlur(section_of_interest, (99, 99), 0)
    image[start_y:end_y, start_x:end_x] = blurred_section
    return image

if(len(sys.argv)<3):
    print("Wrong usage, expected: 'main.py <input_video_path> <output_video_path>'")
    exit(0)

input_video_path = sys.argv[1]
output_video_path = sys.argv[2]

if not os.path.exists(input_video_path):
    print("Input video path not found")
    exit()

out_frames_folder_name = 'output_frames'
if not os.path.exists(out_frames_folder_name):
    os.makedirs(out_frames_folder_name)

codec = cv2.VideoWriter_fourcc(*"XVID")
frame = next(read_frames(input_video_path, 1))
frame_h, frame_w, _ = frame.shape
out = cv2.VideoWriter(output_video_path, codec, 29, (frame_w, frame_h))

num_frames = 10
if(len(sys.argv) == 4):
    num_frames = int(sys.argv[3])

for frame_count, frame in enumerate(read_frames(input_video_path, num_frames)):
    print("FRAME NUM:", frame_count)

    detected_texts = textInImage(frame)
    for c, r, w, h in detected_texts:
        img2=cv2.rectangle(frame, (c,r), (c+w,r+h), (255,0,0), 2) 
        apply_blurring(frame, c, r, c+w, r+h)

    detected_faces = detect_faces(frame)
    for c1, r1, w, h in detected_faces:
        c2 = c1+w
        r2 = r1+h
        frame = cv2.rectangle(frame, (c1,r1), (c2,r2), (0,0,255), 2) 
        frame = apply_blurring(frame, c1, r1, c2, r2)

    adjusted_coordinates = kalman_filter(detected_faces, frame_count)
    for c1, r1, c2, r2 in adjusted_coordinates:
        frame = cv2.rectangle(frame, (c1,r1), (c2,r2), (0,255,0), 2) 
        frame = apply_blurring(frame, c1, r1, c2, r2)

    # filename = f'{out_frames_folder_name}/david_{frame_count}_face_char_kalman_multi.png'
    # cv2.imwrite(filename, img2)
    out.write(frame)
out.release()
