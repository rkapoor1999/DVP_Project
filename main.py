import os
import cv2
import sys
from face_detection_cv2 import detect_faces
from character_detection import textInImage
from kalman_filter import kalman_filter

def read_frames(path, max_frames = -1):
    cap = cv2.VideoCapture(path)
    num_frames = 0
    while True:
        ret, frame = cap.read()
        num_frames+=1
        if not ret:
            break
        if(max_frames>0 and num_frames>max_frames):
            break
        yield frame
    cap.release()

def apply_blurring(image, start_x, start_y, end_x, end_y):
    (h, w) = image.shape[:2]
    start_x = max(0, min(start_x, w))
    start_y = max(0, min(start_y, h))
    end_x = max(0, min(end_x, w))
    end_y = max(0, min(end_y, h))
    
    section_of_interest = image[start_y:end_y, start_x:end_x]
    blurred_section = cv2.GaussianBlur(section_of_interest, (99, 99), 0)
    image[start_y:end_y, start_x:end_x] = blurred_section
    return image

def has_overlap(rec1, rec2):
    max_left = max(rec1[0], rec2[0])
    max_bottom = max(rec1[1], rec2[1])
    min_right = min(rec1[0]+rec1[2], rec2[0]+rec2[2])
    min_top = min(rec1[1]+rec1[3], rec2[1]+rec2[3])
    return (max_left<min_right) and (max_bottom<min_top)


def merge(curr, prev):
    # print(curr, prev)
    curr = sorted(curr, key=lambda box: box[0])
    prev = sorted(prev, key=lambda box: box[0])
    ci = 0
    pi = 0

    result = []
    while(ci<len(curr) and pi<len(prev)):
        if(has_overlap(curr[ci], prev[pi])):
            result.append(curr[ci])
            ci+=1
            pi+=1
        elif(curr[ci][0]<prev[pi][0]):
            result.append(curr[ci])
            ci+=1
        else:
            result.append(prev[pi])
            pi+=1
    while(ci<len(curr)):
        result.append(curr[ci])
        ci+=1
    while(pi<len(prev)):
        result.append(prev[pi])
        pi+=1
    return result






if(len(sys.argv)<3):
    print("Wrong usage, expected: 'main.py <input_video_path> <output_video_path>'")
    exit(0)

input_video_path = sys.argv[1]
output_video_path = sys.argv[2]

if not os.path.exists(input_video_path):
    print("Input video path not found")
    exit()

codec = cv2.VideoWriter_fourcc(*"XVID")
frame = next(read_frames(input_video_path, 1))
frame_h, frame_w, _ = frame.shape
out = cv2.VideoWriter(output_video_path, codec, 29, (frame_w, frame_h))

num_frames = -1
if(len(sys.argv) == 4):
    num_frames = int(sys.argv[3])

prev_faces = []
for frame_count, frame in enumerate(read_frames(input_video_path, num_frames)):
    print("FRAME NUM:", frame_count)

    detected_texts = textInImage(frame)
    for c, r, w, h in detected_texts:
        img2=cv2.rectangle(frame, (c,r), (c+w,r+h), (255,0,0), 2) 
        apply_blurring(frame, c, r, c+w, r+h)

    detected_faces = detect_faces(frame)
    # detected_faces = merge(detected_faces, prev_faces)
    # prev_faces = detected_faces
    for c1, r1, w, h in detected_faces:
        c2 = c1+w
        r2 = r1+h
        frame = cv2.rectangle(frame, (c1,r1), (c2,r2), (0,0,255), 2) 
        frame = apply_blurring(frame, c1, r1, c2, r2)

    adjusted_coordinates = kalman_filter(detected_faces, frame_count)
    for c1, r1, c2, r2 in adjusted_coordinates:
        frame = cv2.rectangle(frame, (c1,r1), (c2,r2), (0,255,0), 2) 
        frame = apply_blurring(frame, c1, r1, c2, r2)

    out.write(frame)
out.release()
