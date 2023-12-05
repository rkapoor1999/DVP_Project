import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    "/Users/kapoor/Desktop/ASU/Notes/Semester 1/DVP/Project/Model/deploy.prototxt",  # Path to the deploy prototxt file
    "/Users/kapoor/Desktop/ASU/Notes/Semester 1/DVP/Project/Model/res10_300x300_ssd_iter_140000.caffemodel"  # Path to the pre-trained model file
)

video_path = "/Users/kapoor/Downloads/MatchConference.mp4"

video_capture = cv2.VideoCapture(video_path)
while True:
    ret, frame = video_capture.read()

    if not ret:
        break
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < .5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
