import cv2
import numpy as np

image = cv2.imread('D:\opencv\cont-2.png')
image=cv2.resize(image,(400,400))

# for lighter images

l_b=np.array([141,223,0])
u_b=np.array([255,255,255])
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)    
mask=cv2.inRange(hsv,l_b,u_b)
thresh=cv2.bitwise_and(image,image,mask=mask)
thresh=cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
#cv2.imshow("thresh",thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
opening_bgr=cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)

#_, threshold = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
  
# using a findContours() function
contours, _ = cv2.findContours(
    opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
black_img=np.zeros((400,400,3))
cv2.drawContours(black_img, contours, -1, (0, 0, 255), 1)

#print(f'No-of contours detected--{len(contours)}')
def find_contour_areas(contours,black_img):
    areas=[]
    char_num=97
    names=[]
    coord=[]
    for cnt,i in zip(contours,range(len(contours))):
        cont_area=cv2.contourArea(cnt)
        areas.append(cont_area)
        M=cv2.moments(cnt)
        cX=int(M["m10"] / M["m00"])
        cY=int(M["m01"] / M["m00"])
        names.append(chr(char_num))
        
        char_num+=1
        coord.append((cX,cY))
        black_img=cv2.circle(black_img,coord[i],3,[255,0,0],2)

    return names,areas,coord
# set target coordinates
target=(100,250)
names,areas,coord=find_contour_areas(contours,black_img)
black_img=cv2.circle(black_img,target,3,[0,255,0],2)
cv2.putText(black_img, 'Target', (target[0]+10,target[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)



def shortest(coord,target,names):
    short=[]
    dist=0
    for ele,ind in zip(coord,range(len(coord))):
        dist=np.sqrt((ele[0]-target[0])**2+(ele[1]-target[1])**2)
        short.append([coord[ind],names[ind],dist])
    return short
dist=shortest(coord,target,names)

dist.sort(key = lambda dist:dist[2])
print(dist)

black_img=cv2.circle(black_img,dist[0][0],3,[255,255,255],2)
cv2.putText(black_img, 'Shortest', (dist[0][0][0]+10,dist[0][0][1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)

cv2.line(black_img,target,dist[0][0],[127,127,127],2)
cv2.imshow('raw_contours',black_img)
cv2.imwrite('distance.jpg',black_img)
  
'''i = 0
  
# list for storing names of shapes
for contour in contours:
  
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue
  
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
      
    # using drawContours() function
    cv2.drawContours(opening_bgr, [contour], 0, (0, 0, 255), 5)
  
    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
  
    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(opening_bgr, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 255, 255), 1)
  
    elif len(approx) == 4:
        cv2.putText(opening_bgr, 'Quadrilateral', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 255, 255), 1)
  
    elif len(approx) == 5:
        cv2.putText(opening_bgr, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 255, 255), 1)
  
    elif len(approx) == 6:
        cv2.putText(opening_bgr, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 255, 255), 1)
  
    else:
        cv2.putText(opening_bgr, 'circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.2, (255, 255, 255), 1)
  
# displaying the image after drawing contours
# cv2.imshow('shapes',opening)'''




'''cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]

blobs = 0
for c in cnts:
    area = cv2.contourArea(c)
    cv2.drawContours(opening_bgr, [c], -1, (36,255,11), -1)
    if area > 13000:
        blobs += 2
    else:
        blobs += 1

print('blobs:', blobs)'''

#cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
#cv2.imshow('opening_bgr', opening_bgr)
cv2.imshow('image', image)
cv2.imwrite('input-img.jpg',image)
#cv2.imshow('mask', mask)

k=cv2.waitKey(0)
if k==ord('q'):
    cv2.destroyAllWindows()