import cv2
import os

#гиперпараметры:
MODEL_TYPE = "faces.xml"
DIR = 'images\\'
FILE = 1
SCL_FACTOR = 1.8
NEIGHBORDS = 2

#----------------------------------------------

files = os.listdir(DIR)

image = cv2.imread(DIR + files[FILE])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

model = cv2.CascadeClassifier(MODEL_TYPE)

objects = model.detectMultiScale(image_gray, scaleFactor = SCL_FACTOR,
                                 minNeighbors = NEIGHBORDS)

for (x, y, w, h) in objects:
    cv2.rectangle(image, (x,y,), (x + w, y + h),
                 (0,255,255), thickness = 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
