import cv2
import os
import pandas as pd



cap = cv2.VideoCapture(0)


pTime = 0
cTime = 0

targets=['square','duck','circle']

y=[]
counter=15
os.chdir(f"codes/{targets[2]}")

while True:
    success, img = cap.read()
    #print(img.shape)
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cv2.imshow("Image", img)
    
    k=cv2.waitKey(1)
    if k==ord('c'):
        #print(pos)
        cv2.imwrite(f"{targets[2]}-{counter+1}.jpg",img)
        counter+=1
        print("saved")
 
        

        

    elif k==ord('q'):
        cv2.destroyAllWindows()
        break