import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
cap=cv.VideoCapture(0) # this 0 ,1 ,-1 etc are teh flags whihc are for webcams or any camera device attached to this system
# instead of the flag you can also put a downloaded video directly  as a file path
fourcc=cv.VideoWriter_fourcc(*'XVID')  # "DIVX", "XVID", "H264", "DX50"  most used compression types  *'xvid' converts it into a list
# of characters
out=cv.VideoWriter('output.avi',fourcc,30,(640,480)) # last two args is kitne frames per sec write karna hai and height,width of each frame
while(cap.isOpened()):  # instead of true in while loop we can use cv.isopened() which returns a boolean if the video capture ka flag
    # is actually present .if wron file location or if wrong cameera attachment which isnt connected then it will return False
    ret,frame=cap.read()   # the firsst value we store is the boolean which shows if the frame is available or not,second is the frame 
    # or we can say frame is the imshow wala args which if the frame is available then shows it to us
    #gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if ret==1:
        #hls=cv.cvtColor(frame,cv.COLOR_BGR2HLS)
        font=cv.FONT_ITALIC
        text=str(datetime.datetime.now())
        frame=cv.putText(frame,text,(10,50),font,1,(127,127,127),2)
        cv.imshow("video",frame)
        #print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        #print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        k=cv.waitKey(1)
        #out.write(frame)
        if k== ord('q'):
            cv.destroyAllWindows()
            break
        else:
            pass
    else:
        break
out.release()  # releases all the resources used for that task
cap.release()
cv.destroyAllWindows()
    
