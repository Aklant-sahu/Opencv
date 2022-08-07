import cv2
'''img=cv2.imread('what.png',0)
cv2.imshow('Output',img)
cv2.waitKey(2000)  # if waitkey==0 then usi code block p[e rhega uske aage code execute nhi hoga
# when ww=aitkey== a certain no then sirf utne time ke liye code freeze hoga uss img pe then aage execution continue
cv2.destroyAllWindows()  # if we want to close all windows automaticlaly then use this
# if we want to close a specific window then use destroywindow(window_name)
img2=cv2.imread('img1.png')
cv2.imshow('Output2',img2)
cv2.waitKey(0)'''

img=cv2.imread('what.png',-1)
img=cv2.resize(img,(500,500))
img=cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Output',img)
k=cv2.waitKey(0)  # waitkey reads the keyboard input and returns that which gets stored in k variable 
# since we know compuiters follow ascii format so all characters are assigned a number whihc the character when pressed 
#gets fetched by waitkey directly from keyboard as a number and are then stored in the variable k
if k==27:
    print('hey ! you pressed ESC so we are closing the window')
    cv2.destroyAllWindows()
elif k==ord('s'):
    print('You pressed s so we are ssaving the img')
    cv2.imwrite('saved_img.png',img)