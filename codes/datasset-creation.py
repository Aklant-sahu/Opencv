import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
path="codes/sl-project/A/"
count=0
capture=np.zeros((480, 640, 3))
cap=cv.VideoCapture(0) # this 0 ,1 ,-1 etc are teh flags whihc are for webcams or any camera device attached to this system
# instead of the flag you can also put a downloaded video directly  as a file path
 # "DIVX", "XVID", "H264", "DX50"  most used compression types  *'xvid' converts it into a list
# of characters
 # last two args is kitne frames per sec write karna hai and height,width of each frame
while(cap.isOpened()):  # instead of true in while loop we can use cv.isopened() which returns a boolean if the video capture ka flag
    # is actually present .if wron file location or if wrong cameera attachment which isnt connected then it will return False
    ret,frame=cap.read()   # the firsst value we store is the boolean which shows if the frame is available or not,second is the frame 
    # or we can say frame is the imshow wala args which if the frame is available then shows it to us
    #gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if ret==1:
        #hls=cv.cvtColor(frame,cv.COLOR_BGR2HLS)
        
        cv.imshow("video",frame)
        cv.imshow("captured",capture)
        #print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        #print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        k=cv.waitKey(1)
        #out.write(frame)
        if k== ord('q'):
            cv.destroyAllWindows()
            break
        elif(k==ord('c')):
            capture=frame
            
            cv.imwrite(path+f"{count}.jpg",capture)
            
            count+=1

    else:
        break

cap.release()
cv.destroyAllWindows()