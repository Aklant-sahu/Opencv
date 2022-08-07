import cv2
import numpy as np
img=cv2.imread('D:\opencv\what.png')

img=cv2.resize(img,(300,300))
img[150:,150:,:]=0
img2=np.zeros(img.shape,dtype='uint8')
# masks are binary images either 0 or 255
bit_and=cv2.bitwise_and(img,img2)
bit_or=cv2.bitwise_or(img,img2)
#when masks are applied on bitwse operations then both the src1 and src must the img1 itself taaki mask ek img1 pe
# apply hoke dusre img1 pe bitwise and opetration hoga

cv2.imshow("img",img)
cv2.imshow("img2",img2)
cv2.imshow("bit_and",bit_and)
cv2.imshow("bit_or",bit_or)
cv2.waitKey(0)