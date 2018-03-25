# Python OpenCV Computer Vision Training
# Module 3: Object Detection

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# img = cv2.imread('./images/detect_blob.png', 0)

# img = Image.open('./images/detect_blob.png')
# img.show()

# height, width = img.shape[0:2]
# cv2.imshow("Original BW",img)
# plt.imshow(img)
# plt.show()

# Manual Thresholding

# threshold = 85
#binary = np.zeros([height,width],'uint8')
# for row in range(0,height):
# 	for col in range(0, width):
# 		if img[row][col]>thresh:
# 			binary[row][col]=255

# plt.imshow(binary,cmap='gray')
# plt.show()
# # cv2.imshow("Slow Binary",binary)

# CV2 Thresholding
# ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,threshold,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO_INV)

# cv2.imshow("CV Threshold",thresh)
# plt.imshow(thresh3,cmap='gray')
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Exercise: Thresholding 

# img = cv2.imread('./images/opencv-logo.png', 0)
# threshold = 150
# ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
# ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
# ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_TRUNC)
# ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO)
# ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO_INV)
# cv2.imshow("CV Threshold",thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Adaptive Thresholding
# img = cv2.imread('./images/sudoku.png',0)
# cv2.imshow("Original",img)

# ret, thresh_basic = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
# plt.imshow(thresh_basic,cmap='gray')
# plt.show()
# cv2.imshow("Basic Binary",thresh_basic)

# thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# plt.imshow(thres_adapt ,cmap='gray')
# plt.show()
# cv2.imshow("Adaptive Threshold",thres_adapt)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Ex: Adaptive Thresholding
# img = cv2.imread('./images/opencv-logo.png', 0)
# thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# cv2.imshow("Adaptive Threshold",thres_adapt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Composite Filtering

# img = cv2.imread('./images/faces.jpeg',1)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h = hsv[:,:,0]
# s = hsv[:,:,1]
# v = hsv[:,:,2]

# hsv_split = np.concatenate((h,s,v), axis=1)
# ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)
# ret, max_hue = cv2.threshold(h,15, 255, cv2.THRESH_BINARY_INV)

# final = cv2.bitwise_and(min_sat,max_hue)
# cv2.imshow("Final",final)
# cv2.imshow("Original",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Ex: Composite Filtering
# img = cv2.imread('./images/tomatoes.jpg',1)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h = hsv[:,:,0]
# s = hsv[:,:,1]
# v = hsv[:,:,2]

# ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)
# ret, max_hue = cv2.threshold(h,15, 255, cv2.THRESH_BINARY_INV)

# final = cv2.bitwise_and(min_sat,max_hue)
# cv2.imshow("Final",final)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Contouring
img = cv2.imread('./images/detect_blob.png',1)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# img2 = img.copy()
# index = -1
# thickness = 4
# color = (255, 0, 255)

# cv2.drawContours(img2, contours, index, color, thickness)
# cv2.imshow("Contours",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("./images/contour.jpg",img2)
# img = Image.open('./images/contour.jpg')
# img.show()

# Ex: Contouring
# img = cv2.imread('./images/opencv-logo.png', 0)
# thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# img2 = img.copy()
# index = -1
# thickness = 4
# color = (0, 0, 0)

# cv2.drawContours(img2, contours, index, color, thickness)
# cv2.imshow("Contours",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("./images/contour.jpg",img2)
# img = Image.open('./images/contour.jpg')
# img.show()

# Compute Area and Perimeter of a contour

# img = cv2.imread('./images/detect_blob.png',1)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# img2 = img.copy()
# index = -1
# thickness = 4
# color = (255, 0, 255)

# objects = np.zeros([img.shape[0], img.shape[1],3], 'uint8')
# for c in contours:
# 	cv2.drawContours(objects, [c], -1, color, -1)

# 	area = cv2.contourArea(c)
# 	perimeter = cv2.arcLength(c, True)

# 	M = cv2.moments(c)
# 	cx = int( M['m10']/M['m00'])
# 	cy = int( M['m01']/M['m00'])
# 	cv2.circle(objects, (cx,cy), 4, (0,0,255), -1)

# 	print("Area: {}, perimeter: {}".format(area,perimeter))

# cv2.imshow("Contours",objects)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("./images/tomatoes.jpg",1)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# res,thresh = cv2.threshold(hsv[:,:,0], 25, 255, cv2.THRESH_BINARY_INV)

# edges = cv2.Canny(img, 100, 70)
# cv2.imshow("Canny",edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("./images/screenshot.jpg",edges)
# img = Image.open('./images/screenshot.jpg')
# img.show()

# Challenge
# import random

# img = cv2.imread("./images/fuzzy.png",1)
# cv2.imshow("Original",img)

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3),0)

# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
# cv2.imshow("Binary",thresh)

# _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))

# filtered = []
# for c in contours:
# 	if cv2.contourArea(c) < 1000:continue
# 	filtered.append(c)

# print(len(filtered))

# objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')
# for c in filtered:
# 	col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
# 	cv2.drawContours(objects,[c], -1, col, -1)
# 	area = cv2.contourArea(c)
# 	p = cv2.arcLength(c,True)
# 	print(area,p)

# cv2.imshow("Contours",objects)
	

# cv2.waitKey(0)
# cv2.destroyAllWindows()
