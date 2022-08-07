import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cap=cv2.VideoCapture('D:\opencv\mall-video-2.mp4')
detector=cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    #frame=cv2.resize(frame,(400,400))
    #print(frame.shape)
    roi=frame[80:,:400,:]
    
    mask=detector.apply(frame)
    _,mask=cv2.threshold(mask,252,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #kernel=np.ones((10,10),np.uint8)
    

    
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours,_=cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
        area=cv2.contourArea(cnt)
        if area>1500:
            #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),[255,0,0],thickness=1)

    #print(mask.shape)
    #thresh=cv2.threshold(mask,0,1,cv2.THRESH_BINARY)
    if ret == True:
        #cv2.imshow('opening',opening)
        cv2.imshow('Frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('roi',roi)

   
    
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    else: 
        break

    
cap.release()    
cv2.destroyAllWindows()