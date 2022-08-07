import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cap=cv.VideoCapture(0)
 
while True:
    ret,frame=cap.read()
    frame=cv.resize(frame,(400,400))
    img=np.zeros((800,800,3),'uint8')# uint means unsingneed int and can accept val between 0-255 wheareas int8 -126 se 127

    img[:400,:400]=frame
    img[:400,400:]=frame
    img[400:,:400]=frame
    img[400:,400:]=frame

    cv.imshow('4-webcam',img)
    if cv.waitKey(1)==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
