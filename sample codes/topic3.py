# Harris Corner Detector

import numpy as np
import cv2 as cv
filename = 'chessboard.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

## result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

## Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

# SIFT (Scale-Invariant Feature Transform)

import numpy as np
import cv2 as cv
img = cv.imread('home.jpg')

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()

kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)

## FAST Algorithm for Corner Detection

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('simple.jpg',0)

## Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

## find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

## Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imwrite('fast_true.png',img2)

## Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imwrite('fast_false.png',img3)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Threhsolding

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv2.imread('./images/detect_blob.png', 0)

height, width = img.shape[0:2]
cv2.imshow("Original BW",img)

# CV2 Thresholding
ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,threshold,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO_INV)

cv2.imshow("CV Threshold",thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Activity: Thresholding 

img = cv2.imread('./images/opencv-logo.png', 0)
threshold = 150
ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_TRUNC)
ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO)
ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO_INV)
cv2.imshow("CV Threshold",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Adaptive Thresholding
img = cv2.imread('./images/sudoku.png',0)
cv2.imshow("Original",img)

ret, thresh_basic = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
plt.imshow(thresh_basic,cmap='gray')
plt.show()
cv2.imshow("Basic Binary",thresh_basic)

thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
plt.imshow(thres_adapt ,cmap='gray')
plt.show()
cv2.imshow("Adaptive Threshold",thres_adapt)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Activity: Adaptive Thresholding
img = cv2.imread('./images/opencv-logo.png', 0)
thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow("Adaptive Threshold",thres_adapt)
cv2.waitKey(0)
cv2.destroyAllWindows()