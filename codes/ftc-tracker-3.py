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



while(cap.isOpened()): 
    ret,frame=cap.read()
    frame_duck=frame
    frame_square=frame
    frame_circle=frame   
    if ret==1:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        l_b_square=np.array([16,142,255])
        u_b_square=np.array([26,255,255])
    
        mask_square=cv2.inRange(hsv,l_b_square,u_b_square)
        res_square=cv2.bitwise_and(frame_square,frame_square,mask=mask_square)

        res_square=cv2.cvtColor(res_square,cv2.COLOR_BGR2GRAY)

        l_b_circle=np.array([0,0,179])
        u_b_circle=np.array([255,23,255])
    
        mask_circle=cv2.inRange(hsv,l_b_circle,u_b_circle)
        res_circle=cv2.bitwise_and(frame_circle,frame_circle,mask=mask_circle)
        
        res_circle=cv2.cvtColor(res_circle,cv2.COLOR_BGR2GRAY)
        gray_circle=cv2.morphologyEx(res_circle, cv2.MORPH_DILATE,kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)

        gray_square=cv2.morphologyEx(res_square, cv2.MORPH_DILATE,kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)
        # gray=cv2.morphologyEx(gray, cv2.MORPH_OPEN,kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)
       

        cnts = cv2.findContours(gray_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for i in range(2):
            for c in cnts:
                cv2.drawContours(gray_square,[c], 0, (255,255,255), -1)

        cnts = cv2.findContours(gray_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for i in range(2):
            for c in cnts:
                cv2.drawContours(gray_circle,[c], 0, (255,255,255), -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        # opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        gray_circle = cv2.morphologyEx(gray_circle, cv2.MORPH_OPEN, kernel, iterations=2)
        contours,_ = cv2.findContours(gray_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            count_square=0
            #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
            area=cv2.contourArea(cnt)
            if area>200:
                count_square+=1
                #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
                x,y,w,h=cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],thickness=1)
                cv2.putText(frame,f"({x},{y})",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,[255,0,0],thickness=1)

        contours,_ = cv2.findContours(gray_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            count_circle=0
            #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
            area=cv2.contourArea(cnt)
            if area>200:
                count_circle+=1
                #cv2.drawContours(frame,[cnt],-1,[255,0,0],1)
                x,y,w,h=cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],thickness=1)
                cv2.putText(frame,f"({x},{y})",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,[255,0,0],thickness=1)
            

        cv2.imshow('gray_square', gray_square)
        cv2.imshow('gray_circle', gray_circle)
        cv2.imshow('frame', frame)
        # print(f"THe total white circular obj are-{count_circle}")
        # print(f"THe total yellow square obj are-{count_square}")
        # cv2.imshow('opening', opening)
        # cv2.imwrite('opening.png', opening)
        
        k=cv2.waitKey(1)
        
        if k== ord('q'):
            cv2.destroyAllWindows()
            break
        
    else:
        break
  # releases all the resources used for that task
cap.release()
cv2.destroyAllWindows()