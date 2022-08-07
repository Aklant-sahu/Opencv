import cv2 as cv2
import numpy as np

   
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('big_buck_bunny_720p_1mb.mp4')
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) # Although you caan put in any values but it will show videos in only
# those resolutions whihc are avilble in your webcam.
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
# Read until video is completed
while(cap.isOpened()):
      
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
   
    # Display the resulting frame
    cv2.imshow('Frame', frame)
   
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   
  # Break the loop
  else: 
    break
   
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()