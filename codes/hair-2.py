import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# image path:    
#path = "D://opencvImages//"
#fileName = "out.jpg"

# Reading an image in default mode:
inputImage = cv2.imread('D:\opencv\hair.png')
inputImage=cv2.resize(inputImage,(400,400))

# Convert RGB to grayscale:
#grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Convert the BGR image to HSV:
hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

# Create the HSV range for the blue ink:
# [128, 255, 255], [90, 50, 70] [10,77,67] [107,145,86]
l_b = np.array([35,83,0])
u_b = np.array([118,255,160])

# Get binary mask of the blue ink:
mask=cv2.inRange(hsvImage,l_b,u_b)
#print(mask.shape)
#ret,res = cv2.threshold(mask,50,255,cv2.THRESH_BINARY_INV)
#print(res.shape)
res=np.array(inputImage)
black=np.array(inputImage)
#res=cv2.bitwise_and(inputImage,inputImage,mask=res)
for i in range(400):
    for j in range(400):
        if mask[i][j]==255:
            res[i][j]=(78,78,101)
for i in range(400):
    for j in range(400):
        if mask[i][j]==255:
            black[i][j]=(49,49,58)
#cv2.imshow('frame',frame)
#cv2.imshow('frame',frame)

dst = cv2.addWeighted(inputImage, 0.7, res, 0.3, 0)
dst1 = cv2.addWeighted(dst, 0.8, black, 0.2, 0)

#img_arr = np.hstack((inputImage, res,dst))
img_arr2 = np.hstack((inputImage, dst,dst1))
cv2.imwrite('D:\opencv\output.png',dst1)
cv2.imwrite('D:\opencv\inputimage.png',inputImage)


cv2.imshow('Input Images',img_arr2)
#cv2.imshow('Input Images',img_arr)
#cv2.imshow('Blended Image',dst)


cv2.imshow('mask',mask)
#cv2.imshow('res',res)
cv2.imshow('res',res)

    
#(T, thresh) = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("Threshold Binary", thresh)
    
    
k=cv2.waitKey(0)
if (k==ord('q')):
    cv2.destroyAllWindows()