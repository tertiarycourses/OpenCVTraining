# Python OpenCV Computer Vision Training
# Module 2: Basic OpenCV Operations

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
# img = cv2.imread("./images/detect_blob.png",1)
# img = img[:,:,::-1]
# plt.imshow(img)
# plt.xticks([]), plt.yticks([])
# plt.show()

# Output image
# cv2.imwrite("./images/output.jpg",img)

# Create image with numpy
img = np.zeros([512,512,3],np.uint8)

# Draw a diagonal blue line with thickness of 5 px
# cv2.line(img,(0,0),(512,512),(255,0,0),5)

# Draw a rectangle
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

# Draw a circle
# cv2.circle(img,(447,63), 63, (0,0,255), -1)

# Draw a polygon
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))

# Add text
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

# cv2.imshow("Draw",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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
# img = cv2.imread("./images/thresh.jpg")
# cv2.imshow("Original",img)

# blur = cv2.GaussianBlur(img, (5,55),0)
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
# 	# Capture frame-by-frame
# 	ret, frame = cap.read()

# 	# frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
# 	cv2.imshow("Frame",frame)

# 	ch = cv2.waitKey(1)
# 	if ch & 0xFF == ord('q'):
# 		break

# cap.release()
# cv2.destroyAllWindows()

# Play video from file
# cap = cv2.VideoCapture('./images/video.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# Save Video
# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)

#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# Release everything if job is finished
# cap.release()
# out.release()
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

# Bind Mouse Callback Function

# mouse callback function
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)

# # Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# Drawing Canvas - 1
# drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
# ix,iy = -1,-1

# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv2.circle(img,(x,y),5,(0,0,255),-1)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv2.circle(img,(x,y),5,(0,0,255),-1)

# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break

# cv2.destroyAllWindows()

# Drawing Canvas - 2
# canvas = np.ones([500,500,3],'uint8')*255
# radius = 3
# color = (0,255,0)
# pressed = False

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