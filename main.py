import os
import cv2
import sys
from detect_faces_dlib import detect_faces, detect_faces_tolist
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
    section_of_interest = image[start_y:end_y, start_x:end_x]
    blurred_section = cv2.GaussianBlur(section_of_interest, (15, 15), 0)
    image[start_y:end_y, start_x:end_x] = blurred_section

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
for frame_count, frame in enumerate(read_frames(input_video_path, num_frames)):
    print("FRAME NUM:", frame_count)

    detected_texts = textInImage(frame)
    for c, r, w, h in detected_texts:
        img2=cv2.rectangle(frame, (c,r), (c+w,r+h), (255,0,0), 2) 
        apply_blurring(frame, c, r, c+w, r+h)
        
    detected_faces = detect_faces_tolist(frame)
    for c, r, w, h in detected_faces:
        img2=cv2.rectangle(frame, (c,r), (c+w,r+h), (0,0,255), 2) 
        apply_blurring(frame, c, r, c+w, r+h)

    adjusted_coordinates = kalman_filter(detected_faces, frame_count)
    for c1, r1, c2, r2 in adjusted_coordinates:
        pass
        img2=cv2.rectangle(frame, (c1,r1), (c2,r2), (0,255,0), 2) 
        apply_blurring(frame, c1, r1, c2, r2)

    filename = f'{out_frames_folder_name}/david_{frame_count}_face_char_kalman_multi.png'
    cv2.imwrite(filename, img2)
    out.write(img2)
out.release()
