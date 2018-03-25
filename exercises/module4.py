import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Template Matching
# template = cv2.imread('./images/template.jpg',0)
# frame = cv2.imread("./images/players.jpg",0)

# cv2.imshow("Frame",frame)
# cv2.imshow("Template",template)

# result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# print(max_val,max_loc)
# cv2.circle(result,max_loc, 15,255,2)

# cv2.imwrite("./images/screenshot.jpg",result)
# img = Image.open('./images/screenshot.jpg')
# img.show()

# cv2.imshow("Matching",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Haar Cascade Face Detection

# img = cv2.imread("./images/faces.jpeg",1)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# path = "haarcascade_frontalface_default.xml"

# face_cascade = cv2.CascadeClassifier(path)

# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
# print(len(faces))

# for (x, y, w, h) in faces:
# 	cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

# cv2.imwrite("./images/screenshot.jpg",img)
# img = Image.open('./images/screenshot.jpg')
# img.show()

# cv2.imshow("Image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img = cv2.imread("./images/faces.jpeg",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path = "haarcascade_eye.xml"

eye_cascade = cv2.CascadeClassifier(path)

eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.02,minNeighbors=20,minSize=(10,10))
print(len(eyes))

for (x, y, w, h) in eyes:
	xc = (x + x+w)/2
	yc = (y + y+h)/2
	radius = w/2
	cv2.circle(img, (int(xc),int(yc)), int(radius), (255,0,0), 2)

cv2.imwrite("./images/screenshot.jpg", img)
img = Image.open('./images/screenshot.jpg')
img.show()

# cv2.imshow("Eyes",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()