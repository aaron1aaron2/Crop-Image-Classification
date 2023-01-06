import cv2
import numpy as np

## Read
# img = cv2.imread("notebook/image/2a23c0c3-b2c2-46dc-8a5e-923a196bee99.jpg")
img = cv2.imread("notebook/image/0d9eac13-87b6-4449-98cb-59d3ccaedbc3_resize.jpg")


## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
mask = cv2.inRange(hsv, (10, 0, 0), (80, 255,255))

## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]
cv2.imshow('test', green)
cv2.waitKey(0)
cv2.destroyAllWindows()

(mask == 255).sum()

## save 
# cv2.imwrite("green.png", green)
