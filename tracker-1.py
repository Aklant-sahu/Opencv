import cv2
import numpy as np
def nothing(x):  # thsi function can be anything and this is a callback func wwhich gets called when trackbar
    #position changes..this is a compulsory callback function so we should make this function.This function always takes the current 
    # trackbar pos as an input so make sure to include it.
    pass

img=np.zeros((400,400,3),dtype='uint8')
cv2.namedWindow('image')
cv2.createTrackbar('B',"image",0,255,nothing)
cv2.createTrackbar('G',"image",0,255,nothing)
cv2.createTrackbar('R',"image",0,255,nothing)

while(1):
    b=cv2.getTrackbarPos('B','image')
    g=cv2.getTrackbarPos('G','image')
    r=cv2.getTrackbarPos('R','image')
    img[:,:,:]=(b,g,r)
    cv2.imshow("image",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        cv2.destroyAllWindows()
        break


