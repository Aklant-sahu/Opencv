import cv2

image = cv2.imread('2022-07-01.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    cv2.drawContours(gray,[c], 0, (255,255,255), -1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

cv2.imshow('gray', gray)
cv2.imshow('opening', opening)
cv2.imwrite('opening.png', opening)
cv2.waitKey(0)