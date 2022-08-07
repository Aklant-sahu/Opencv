import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cap=cv2.VideoCapture(0) 
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
detector=cv2.createBackgroundSubtractorMOG2()

def nothing(x):
    pass

while (cap.isOpened()):
    ret,frame=cap.read()
    #frame=cv2.resize(frame,(400,400))
    #print(frame.shape)
    
    
    mask=detector.apply(frame)
    _,mask=cv2.threshold(mask,252,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #kernel=np.ones((10,10),np.uint8)
    

    
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours,_=cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
        area=cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],thickness=1)

    print(mask.shape)
    thresh=cv2.threshold(mask,0,1,cv2.THRESH_BINARY)
    if ret == True:
        #cv2.imshow('opening',opening)
        cv2.imshow('Frame', frame)
        cv2.imshow('mask', mask)
        # cv2.imshow('roi',roi)

   
    
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    else: 
        break

    
cap.release()    
cv2.destroyAllWindows()