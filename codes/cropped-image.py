import cv2
import numpy as np
img=cv2.imread('D:/opencv/plot_sample_[39].png')
cropped_img=img[113:176,191:250,:]

cropped_img=cv2.resize(cropped_img,(300,300),cv2.INTER_CUBIC)
cv2.imshow("cropped_img",cropped_img)
cv2.imshow("img",img)
cv2.imwrite('cropped_img.jpg',cropped_img)
cv2.waitKey(0)