import cv2
import numpy as np

image = cv2.imread('D:\opencv\crater.png')
image=cv2.resize(image,(400,400))

# for lighter images

l_b=np.array([0,0,138])
u_b=np.array([255,255,255])
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)    
mask=cv2.inRange(hsv,l_b,u_b)
thresh=cv2.bitwise_and(image,image,mask=mask)
thresh=cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
cv2.imshow("thresh",thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)


cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]

blobs = 0
for c in cnts:
    area = cv2.contourArea(c)
    cv2.drawContours(mask, [c], -1, (36,255,12), -1)
    if area > 13000:
        blobs += 2
    else:
        blobs += 1

print('blobs:', blobs)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imwrite('light-blob.jpg',opening)

cv2.imshow('image', image)
cv2.imwrite('light-img.jpg',image)
cv2.imshow('mask', mask)
cv2.imwrite('light-mask.jpg',mask)

k=cv2.waitKey(0)
if k==ord('q'):
    cv2.destroyAllWindows()