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



# cv2.namedWindow('Tracking')
# cv2.createTrackbar('LH','Tracking',0,255,nothing)
# cv2.createTrackbar('LS','Tracking',0,255,nothing)
# cv2.createTrackbar('LV','Tracking',0,255,nothing)
# cv2.createTrackbar('UH','Tracking',255,255,nothing)
# cv2.createTrackbar('US','Tracking',255,255,nothing)
# cv2.createTrackbar('UV','Tracking',255,255,nothing)

while(cap.isOpened()): 
    ret,frame=cap.read()
    frame_duck=frame
    frame_square=frame
    frame_circle=frame   
    if ret==1:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # l_h=cv2.getTrackbarPos('LH','Tracking')
        # l_s=cv2.getTrackbarPos('LS','Tracking')
        # l_v=cv2.getTrackbarPos('LV','Tracking')
        # u_h=cv2.getTrackbarPos('UH','Tracking')
        # u_s=cv2.getTrackbarPos('US','Tracking')
        # u_v=cv2.getTrackbarPos('UV','Tracking')
        l_b_duck=np.array([21,100,147])
        u_b_duck=np.array([79,255,255])
    
        mask_duck=cv2.inRange(hsv,l_b_duck,u_b_duck)
        res_duck=cv2.bitwise_and(frame_duck,frame_duck,mask=mask_duck)

        l_b_square=np.array([0,211,0])
        u_b_square=np.array([255,255,255])
    
        mask_square=cv2.inRange(hsv,l_b_square,u_b_square)
        res_square=cv2.bitwise_and(frame_square,frame_square,mask=mask_square)

        l_b_circle=np.array([62,0,164])
        u_b_circle=np.array([255,255,255])
    
        mask_circle=cv2.inRange(hsv,l_b_circle,u_b_circle)
        res_circle=cv2.bitwise_and(frame_circle,frame_circle,mask=mask_circle)
        
        res_duck=cv2.cvtColor(res_duck,cv2.COLOR_BGR2GRAY)
        res_square=cv2.cvtColor(res_square,cv2.COLOR_BGR2GRAY)
        res_circle=cv2.cvtColor(res_circle,cv2.COLOR_BGR2GRAY)
        
        res=cv2.bitwise_or(res_duck,res_circle)
        res=cv2.bitwise_or(res,res_square)
    #     kernel=[[0, 0, 1, 0, 0],[1, 1, 1, 1, 1],
    #    [1, 1, 1, 1, 1],
    #    [1, 1, 1, 1, 1],
    #    [0, 0, 1, 0, 0]]
        opening = cv2.morphologyEx(res, cv2.MORPH_OPEN,kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        contours,_=cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            count=0
            #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
            area=cv2.contourArea(cnt)
            if area>200:
                count+=1
                #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
                x,y,w,h=cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],thickness=1)
                cv2.putText(frame,f"({x},{y})",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,[255,0,0],thickness=1)
        # cv2.imshow('res_duck',res_duck)
        # cv2.imshow('res_square',res_square)
        # cv2.imshow('res_circle',res_circle)
        cv2.imshow('res',res)
        print(count)
        cv2.imshow('finalres',opening)
        cv2.imshow('frame',frame)

        

        
        k=cv2.waitKey(1)
        
        if k== ord('q'):
            cv2.destroyAllWindows()
            break
        
    else:
        break
  # releases all the resources used for that task
cap.release()
cv2.destroyAllWindows()