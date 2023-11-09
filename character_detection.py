import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

def textInImage(img):
    image_path = img
    img = cv2.imread(image_path)

    # instanace text detector
    reader = easyocr.Reader(["en"],gpu=False)

    #detect text on image
    text_ = reader.readtext(img)
    threshold = 0.01

    coordinates = []

    for t_,t in enumerate(text_):
        bbox , text, score = t
        bbox = [[int(i) for i in sublist] for sublist in bbox]
        x, y = bbox[0]
        w, h = q[1][0]-q[0][0], q[2][1]-q[1][1]
        if score > threshold:
            coordinates.append([x,y,w,h])
            #cv2.rectangle(img,(x, y), (x+w, y+h),(0,255,0),5)
    #plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #plt.show()

    return coordinates

def main():
    textInImage("/Users/kapoor/Desktop/ASU/Notes/Semester 1/DVP/Assignment 3/MSSIM.png")

if __name__ == '__main__':
    main()