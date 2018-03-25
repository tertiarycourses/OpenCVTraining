import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

# Read image 
# img = cv2.imread("./images/detect_blob.png",1)
# img = cv2.imread("./images/detect_blob.png",0)
# img = cv2.imread("./images/detect_blob.png",-1)

# Display image
# cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
# cv2.imshow("Image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Image attributes
# print(img.shape)
# print(img[10, 5])

# Display image with PIL
# img = Image.open('./images/detect_blob.png')
# img.show()

# Display image with Matplotlib
# img = Image.open('./images/detect_blob.png')
# plt.imshow(img)
# plt.show()

# Output image
# cv2.imwrite("./images/output.jpg",img)

# Create image with numpy
# img = np.zeros([150,200,1],'uint8')
# cv2.imshow("Black",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(img[0,0,:])

# Exercise - Create White and Blue images
# white = np.ones([150,200,3],'uint8')
# white *= (2**16-1)
# cv2.imshow("White",white)
# print(white[0,0,:])

# color = np.ones([150,200,3],'uint8')
# color[:,:] = (255,0,0)
# cv2.imshow("Blue",color)
# print(color[0,0,:])

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Split into BGR channels
# img = cv2.imread("./images/butterfly.jpg", 1)

# Direct method
# b = img[:,:,0]
# g = img[:,:,1]
# r = img[:,:,2]

# CV 2 Split method
# b,g,r = cv2.split(img)
# rgb_split = np.concatenate((r,g,b),axis=1)

# cv2.imshow("RGB",rgb_split)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Split into HSV channels
# img = cv2.imread("./images/butterfly.jpg", 1)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(hsv)
# hsv_split = np.concatenate((h,s,v),axis=1)

# cv2.imshow("HSV",hsv_split)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Ex: Split into BGR and HSV channels
# img = cv2.imread("./images/tomatoes.jpg", 1)

# b,g,r = cv2.split(img)
# rgb_split = np.concatenate((r,g,b),axis=1)
# cv2.imshow("RGB",rgb_split)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(hsv)
# hsv_split = np.concatenate((h,s,v),axis=1)
# cv2.imshow("HSV",hsv_split)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert to Gray Scale image
# img = cv2.imread("./images/butterfly.jpg",1)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow("Gray Scale",gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Add Transparency
# img = cv2.imread("./images/butterfly.jpg",1)
# b = img[:,:,0]
# g = img[:,:,1]
# r = img[:,:,2]

# rgba = cv2.merge((b,g,r,g))
# cv2.imwrite("./images/rgba.png",rgba)

# Exercise: Transparency
# img = cv2.imread("./images/tomatoes.jpg", 1)

# b = img[:,:,0]
# g = img[:,:,1]
# r = img[:,:,2]

# rgba = cv2.merge((b,g,r,g))
# cv2.imwrite("./images/rgba2.png",rgba)

# Blur filter
img = cv2.imread("./images/thresh.jpg")
# cv2.imshow("Original",img)

# blur = cv2.GaussianBlur(img, (5,55),0)
# plt.imshow(blur)
# plt.show()
# cv2.imshow("Blur",blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Dilate filter
# kernel = np.ones((5,5),'uint8')
# dilate = cv2.dilate(img,kernel,iterations=0)
# cv2.imshow("Dilate",dilate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Erode filter
# kernel = np.ones((5,5),'uint8')
# erode = cv2.erode(img,kernel,iterations=1)
# cv2.imshow("Erode",erode)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Resize 
# img = cv2.imread("./images/players.jpg",1)
# img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
# cv2.imshow("Resize",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Stretch
# img = cv2.imread("./images/players.jpg",1)
# img_stretch = cv2.resize(img, (1000,1000))
# img_stretch_near = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_NEAREST)
# cv2.imshow("Stretch",img_stretch)
# cv2.imshow("Stretch near",img_stretch_near)
# ch = cv2.waitKey(0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Rotation
# M = cv2.getRotationMatrix2D((0,0), -30, 1)
# rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow("Rotated",rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Video capture
# cap = cv2.VideoCapture(0)

# while(True):
# 	ret, frame = cap.read()

# 	frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
# 	cv2.imshow("Frame",frame)

# 	ch = cv2.waitKey(1)
# 	if ch & 0xFF == ord('q'):
# 		break

# cap.release()
# cv2.destroyAllWindows()


# # Callback Event
# cap = cv2.VideoCapture(0)

# color = (0,255,0)
# line_width = 3
# radius = 100
# point = (0,0)

# def click(event, x, y, flags, param):
# 	global point, pressed
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		print("Pressed",x,y)
# 		point = (x,y)

# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame",click)

# while(True):
# 	ret, frame = cap.read()

# 	frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
# 	cv2.circle(frame, point, radius, color, line_width)
# 	cv2.imshow("Frame",frame)

# 	ch = cv2.waitKey(1)
# 	if ch & 0xFF == ord('q'):
# 		break

# cap.release()
# cv2.destroyAllWindows()


# Canvas
# Global variables
# canvas = np.ones([500,500,3],'uint8')*255
# radius = 3
# color = (0,255,0)
# pressed = False

# click callback
# def click(event, x, y, flags, param):
# 	global canvas, pressed
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		pressed = True
# 		cv2.circle(canvas,(x,y),radius,color,-1)
# 	elif event == cv2.EVENT_MOUSEMOVE and pressed == True:
# 		cv2.circle(canvas,(x,y),radius,color,-1)
# 	elif event == cv2.EVENT_LBUTTONUP:
# 		pressed = False

# # window initialization and callback assignment
# cv2.namedWindow("canvas")
# cv2.setMouseCallback("canvas", click)

# # Forever draw loop
# while True:

# 	cv2.imshow("canvas",canvas)

# 	# key capture every 1ms
# 	ch = cv2.waitKey(1)
# 	if ch & 0xFF == ord('q'):
# 		break
# 	elif ch & 0xFF == ord('b'):
# 		color = (255,0,0)
# 	elif ch & 0xFF == ord('g'):
# 		color = (0,255,0)
	
# cv2.destroyAllWindows()