import cv2 as cv
img=cv.imread('what.png')

line=cv.line(img,[0,0],[255,255],[10,125,200],10)
arrow=cv.arrowedLine(img,[255,0],[255,255],[10,0,0],10)  ## ye sab jo bhi shapes ho voh directly mere main image pe banenge ie img
# so after these shapes if i directly also show img only then also ill get all those shapes
#  rectangle
# circle we can also draw these shapes
font=cv.FONT_HERSHEY_COMPLEX
text=cv.putText(img,'olivio',[200,500],font,5,[125,125,125],10)
cv.imshow('Output',img)
print(img.shape)
cv.waitKey(0)
