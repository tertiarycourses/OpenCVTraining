import cv2 as cv
import numpy as np
import sys

# Test Image

img = cv.imread("./images/butterfly.jpg")
# img = cv.imread("./images/butterfly.jpg",0)  # Convert to Gray Scale

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv.imshow("Display window", img)

k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("./images/starry_night.png", img)
cv.destroyAllWindows()


# Test Video

cap = cv.VideoCapture(0)
while True:
    ## Capture frame-by-frame
    ret, frame = cap.read()
    ## Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ## Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

## When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# Play Video

cap = cv.VideoCapture('./videos/sample-video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

# Save Video

ap = cv.VideoCapture(0)
while True:
    ## Capture frame-by-frame
    ret, frame = cap.read()
    ## Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ## Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# Create a black image

img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px

cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a diagonal blue line with thickness of 5 px

cv.line(img,(0,0),(512,512),(255,0,0),5)
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a rectangle

cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a circle

cv.circle(img,(447,63), 63, (0,0,255), -1)
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Draw a polygon

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Add text

font = cv2.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Activity: Drawing Function

img = cv.imread("./images/starry_night.jpg")

cv.rectangle(img,(120,150),(160,180),(0,255,0),3)
cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Activity: Mouse Control

drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
## mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)

 ## Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()

# Accessing and Modifying Pixels

img = cv.imread("./images/butterfly.jpg")
k = cv.waitKey(0)
px = img[100,100]
print( px )
img[100,100] = [255,255,255]
px = img[100,100]
print( px )

# Convert BRG to Gray Scale image

img = cv.imread("./images/butterfly.jpg")
gray = cv.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv.imshow("Gray Scale",gray)
cv.waitKey(0)
cv.destroyAllWindows()

# Convert BRG to Gray Scale image

img = cv.imread("./images/butterfly.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv.imshow("HSV",hsv)
cv.waitKey(0)
cv.destroyAllWindows()

# Splitting BGR Channels

img = cv.imread("./images/starry_night.jpg")
cv.imshow("Original", img)

b,g,r = cv.split(img)
cv.imshow("Blue", b)
cv.imshow("Green ", g)
cv.imshow("Red", r)

# Splitting HSV Channels

img = cv.imread("./images/starry_night.jpg")
cv.imshow("Original", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv.imshow("HSV",hsv)

h,s,v = cv2.split(hsv)
cv.imshow("Hue", h)
cv.imshow("Saturation ", s)
cv.imshow("Value", v)

# Merging Channels

img = cv.imread("./images/butterfly.jpg")

cv.imshow("Original", img)

b,g,r = cv.split(img)

img2 = cv.add(b,g,r)
cv.imshow("Merge", img2)

cv.waitKey(0)
cv.destroyAllWindows()

# Activity: Splitting and Merging Channels

img = cv.imread("./images/starry_night.jpg")
cv.imshow("Original", img)

b,g,r = cv.split(img)
cv.imshow("Blue", b)
cv.imshow("Green ", g)
cv.imshow("Red", r)

img2 = cv.merge((b,g,r))
cv.imshow("Merged", img2)

img[:,:,2]=0
cv.imshow("Modified", img)

cv.waitKey(0)
cv.destroyAllWindows()

# Image Addition

img = cv.imread("./images/butterfly.jpg")

cv.imshow("Original", img)

b,g,r = cv.split(img)

img2 = cv.add(b,g,r)
cv.imshow("Add", img2)

cv.waitKey(0)
cv.destroyAllWindows()

# Image Blending

img = cv.imread("./images/butterfly.jpg")
img2 = cv.imread('./images/opencv-logo.png')

dst = cv.addWeighted(img1,0.7,img2,0.3,0)
cv.imshow('dst',dst)

cv.waitKey(0)
cv.destroyAllWindows()

# Scaling

import numpy as np
import cv2 as cv

img = cv.imread('./images/butterfly.jpg')
cv.imshow("Original", img)
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
cv.imshow("Scaled", res)

cv.waitKey(0)
cv.destroyAllWindows()

# Translation

import numpy as np
import cv2 as cv

img = cv.imread('./images/butterfly.jpg')
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)

cv.waitKey(0)
cv.destroyAllWindows()

# Rotation

img = cv.imread('./images/butterfly.jpg',0)
rows,cols = img.shape

# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Affline Transformation
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./images/butterfly.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

# Perspective Transformation

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('sudoku.png')
rows,cols,ch = img.shape

rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

# 2D Convolution ( Image Filtering )

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/butterfly.jpg')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# Averaging Blurring

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('opencv-logo-white.png')
blur = cv.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# Guassian Blurring

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('opencv-logo-white.png')
blur = cv.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# Erosion

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("/home/pi/mu_code/topic2/images/starry_night.jpg")

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(erosion),plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.show()

# Dilation

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


