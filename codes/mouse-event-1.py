import cv2
print([i for i in dir(cv2) if 'EVENT' in i])
def callback1(event,x,y,flags,param):  # This is the specific format with the specific parameters it take in order to create a callback
    if event==cv2.EVENT_LBUTTONDOWN:
        font=cv2.FONT_ITALIC
        text =f'X:{x} Y:{y}'
        cv2.putText(img,text,(x,y),font,0.5,(0,0,0),2) # since it is a callback function so i cant reassign this to img and show 
        # as there is noyhing as img as we are not sending the imaage arg
        cv2.imshow("image",img)
    if event==cv2.EVENT_RBUTTONDOWN:
        font=cv2.FONT_ITALIC
        text =f'[{img[y,x,0]},{img[y,x,1]},{img[y,x,2]}]'
        cv2.putText(img,text,(x,y),font,0.5,(0,0,0),2) # since it is a callback function so i cant reassign this to img and show 
        # as there is noyhing as img as we are not sending the imaage arg
        cv2.imshow("image",img)


img=cv2.imread('D:\opencv\plot_sample_[39].png',1)
#img=cv2.resize(img,(400,400))
cv2.imshow("image",img)
cv2.setMouseCallback("image",callback1)
k=cv2.waitKey(0)
if k==ord('q'):
    cv2.destroyAllWindows()