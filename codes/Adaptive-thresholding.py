import cv2
import numpy as np
'''
Types of common global thresholding techniques
global thresholding cause that threshold is applied to all image pixels and not on specific regions
if a coolor image is passed to thresh binary then each channel value is checked if its equal to or less than threshold thus 
returning a color image
cv2.thresh_binary
cv2.thresh_binary_inv
cv2.thresh_trunc-->jo bhi threshold value se upar voh truncated or written as zero
cv2.thresh_tozero
cv2.thresh_zero_inv

All of these cv2.  are flags or basically simple functions performing matrix transformations
that take image as inp process and give output

Dynamic thresholding is used in a image when lighting conditins vary throughout the image
At that time only a single threshold cant work the whole image .We have to go region wise and we have to set dynamic threshold for 
the image to tackle uneven lighting conditions.

# Adaptive thresholding  only works on gray scale image
'''
img=cv2.imread('D:\opencv\cont-2.png',0)

img=img.astype(np.uint8)
_,th=cv2.threshold(img,200,255,cv2.THRESH_BINARY)

th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,2)
cv2.imshow("img",img)
cv2.imshow("th",th)
cv2.imshow("th2",th2)
cv2.waitKey(0)
