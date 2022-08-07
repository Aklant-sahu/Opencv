import cv2
import numpy as np
img=cv2.imread('D:\opencv\images.jpg',1)

# for darker images
l_b=np.array([0,0,241])
u_b=np.array([40,255,255])
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)    
mask=cv2.inRange(hsv,l_b,u_b)

#mask = cv2.bitwise_not(mask)
masked=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("masked",masked)
gray=cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
cv2.imshow("og_gray",gray)
gray=gray[gray>0]


print(np.max(gray))
print(np.min(gray))
temp=18.8+gray*(37-18.8)/255
print(np.average(temp))
#cv2.imshow("gray",gray)

cv2.waitKey(0)