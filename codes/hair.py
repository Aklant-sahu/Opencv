import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def nothing(x):
    pass


#img=cv2.resize(img,(500,500))
#img=cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
cv2.namedWindow('Tracking') # creating a window named tracking
cv2.createTrackbar('LH','Tracking',0,255,nothing)
cv2.createTrackbar('LS','Tracking',0,255,nothing) # adding sliders to the window named trackbar where first value sshows the 
# strating position of the lsider whereas the second shows till where the slider would slide till what value.
cv2.createTrackbar('LV','Tracking',0,255,nothing)
cv2.createTrackbar('UH','Tracking',255,255,nothing)
cv2.createTrackbar('US','Tracking',255,255,nothing)
cv2.createTrackbar('UV','Tracking',255,255,nothing)
while True:
    frame=cv2.imread('hand.jpg',1)

    frame=cv2.resize(frame,(400,400))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    l_h=cv2.getTrackbarPos('LH','Tracking')  # here we are getting yhe current position of the trackbar to see on which value it is
    # currently at so that we could use it as a bound for determining the mask
    l_s=cv2.getTrackbarPos('LS','Tracking')
    l_v=cv2.getTrackbarPos('LV','Tracking')
    u_h=cv2.getTrackbarPos('UH','Tracking')
    u_s=cv2.getTrackbarPos('US','Tracking')
    u_v=cv2.getTrackbarPos('UV','Tracking')
    l_b=np.array([l_h,l_s,l_v])
    u_b=np.array([u_h,u_s,u_v])
    '''l_b=np.array([16,151,0])
    u_b=np.array([102,255,255])'''
    
    mask=cv2.inRange(hsv,l_b,u_b) # mask is always made using hsv image with theh hsv lower and upper bounds for the task 
    # of object tracking caus eonly in this we can seggregate different colors and build a mask out of it .
    res=cv2.bitwise_and(frame,frame,mask=mask) # this maask is then applie dto the original image where the white portion of the mask
    # is only displayed and rest of the portion is blacked out shoowing we have selected the masked portion.
    #print(1)
    cv2.imshow('frame',frame)
    #cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    #(T, thresh) = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Threshold Binary", thresh)
    
    
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

cv2.destroyAllWindows() 