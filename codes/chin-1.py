import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

img=cv.imread('what.png')
# moving the image
def translate(img,x,y):
    trans=np.float32([[1,0,x],[0,1,y]])
    dimension=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,trans,dimension)

#ans=translate(img,100,-50)
#cv.imshow("ans",ans)
#cv.waitKey(0)

## rotation
def rotate(img,angle,rotationpoint):
    (height,width)=img.shape[:2]
    if rotationpoint==None:
        rotationpoint=(width//2,height//2)
    
    rotmat=cv.getRotationMatrix2D(rotationpoint,angle,1.0)
    dimension=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,rotmat,dimension)

rot=rotate(img,45,None)
#cv.imshow("rot",rot)
#cv.waitKey(0)

# flip
flip=cv.flip(img,1)
#cv.imshow("flip",flip)
#cv.waitKey(0)

# cropped images
print(img.shape)
x=img[500:,500:,:]
print(x.shape)
#cv.imshow("crop",x)
#cv.waitKey(0)

# bgr to gray
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("gray",gray)

#cv.imshow("img",img)
#cv.waitKey(0)   # jitne bhi isse pehle imshow hai jo li destroyed nhi hai voh sab show ho jaega mtlab multiple
# window ek sath

# HUE  
# hsv is hue saturation value which is for showing hue
hue=cv.cvtColor(img,cv.COLOR_BGR2HSV)
#cv.imshow("hue",hue)

#cv.imshow("img",img)
#cv.waitKey(0)            

# l.a.b
lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
#cv.imshow("lab",lab)

#cv.imshow("img",img)
#cv.waitKey(0)

#bgr to RGB

# merge
blank=np.zeros(img.shape[:2],dtype='uint8')
#cv.imshow('blank',blank)
b,g,r=cv.split(img) 
#cv.imshow("blue",b)  # it shows img in intensity channel
blue=cv.merge([b,blank,blank]) # shows blue color in blue not in intensity
#cv.imshow("blue",blue)
#cv.waitKey(0)  

# gaussian blue
gauss=cv.GaussianBlur(img,(5,5),0)
#cv.imshow("gauss",gauss)

# average
avg=cv.blur(img,(3,3))
#cv.imshow("avg",avg)
#cv.waitKey(0)
# median blue
# bilateral blurring   edges are intact centre is blurred
bil=cv.bilateralFilter(img,5,100,100)
cv.imshow("bilateralblur",bil)
cv.imshow("img",img)
cv.waitKey(2000)
cv.destroyAllWindows()
# bitwise operations
cv.imshow('blank',blank)
rect=cv.rectangle(blank.copy(),(30,30),(300,300),255,-1)
circ=cv.circle(blank.copy(),(100,100),(200),255,-1)
cv.imshow('circle',circ)
cv.imshow('rect',rect)
andd=cv.bitwise_and(rect,circ)
cv.imshow("and",andd)
orr=cv.bitwise_or(rect,circ)
cv.imshow("or",orr)
xor=cv.bitwise_xor(rect,circ)
cv.imshow('xor',xor)
cv.waitKey(1000)
cv.destroyAllWindows()

#MASKING
rect=cv.rectangle(blank.copy(),(30,30),(300,300),255,-1)
masked=cv.bitwise_and(img,img,mask=rect)
cv.imshow("masked",masked)
cv.imshow("img",img)
cv.waitKey(2000)
cv.destroyAllWindows()

# image histogram (pixel distribution)
b,g,r=cv.split(img)  # obtaining individual b,g,r channels from the image
print(b.shape)
print(b.reshape(640000))
sns.kdeplot(b.reshape(640000),color='b')
sns.kdeplot(g.reshape(640000),color='g')
sns.kdeplot(r.reshape(640000),color='r')



plt.show()
bins= np.linspace(0,255,10)
plt.hist(b, bins=bins, edgecolor="k")
plt.show()
