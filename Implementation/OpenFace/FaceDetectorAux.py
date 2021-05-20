import cv2

import numpy as np
import imutils
import time
import dlib
from PIL import Image
import cv2

def detect_faces_haar(frame, face_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
        cv2.imshow('image', frame)
        cv2.waitKey()

        #count += 1
        #print(count)
        # Save the captured image into the datasets folder
        #path="images/input" + str(count) + ".jpg"
        #print(path)
        #cv2.imwrite("images/input" + str(count) + ".jpg", img[y1:y2,x1:x2])
        
        