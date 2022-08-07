import cv2 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cap=cv2.VideoCapture(0) 
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

def nothing(x):
    pass



cv2.namedWindow('Tracking')
cv2.createTrackbar('LH','Tracking',0,255,nothing)
cv2.createTrackbar('LS','Tracking',0,255,nothing)
cv2.createTrackbar('LV','Tracking',0,255,nothing)
cv2.createTrackbar('UH','Tracking',255,255,nothing)
cv2.createTrackbar('US','Tracking',255,255,nothing)
cv2.createTrackbar('UV','Tracking',255,255,nothing)

while(cap.isOpened()): 
    ret,frame=cap.read()   
    if ret==1:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        l_h=cv2.getTrackbarPos('LH','Tracking')
        l_s=cv2.getTrackbarPos('LS','Tracking')
        l_v=cv2.getTrackbarPos('LV','Tracking')
        u_h=cv2.getTrackbarPos('UH','Tracking')
        u_s=cv2.getTrackbarPos('US','Tracking')
        u_v=cv2.getTrackbarPos('UV','Tracking')
        l_b=np.array([l_h,l_s,l_v])
        u_b=np.array([u_h,u_s,u_v])
    
        mask=cv2.inRange(hsv,l_b,u_b)
        res=cv2.bitwise_and(hsv,hsv,mask=mask)

        mask1=cv2.inRange(frame,l_b,u_b)
        res1=cv2.bitwise_and(frame,frame,mask=mask)
        
        cv2.imshow('res',res)
        cv2.imshow('res1',res1)
        k=cv2.waitKey(1)
        
        if k== ord('q'):
            cv2.destroyAllWindows()
            break
        
    else:
        break
  # releases all the resources used for that task
cap.release()
cv2.destroyAllWindows()