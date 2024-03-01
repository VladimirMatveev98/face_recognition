import cv2
import os

#гиперпараметры:
MODEL_TYPE = "faces.xml" #Тип модели для распознавания.
DIR = 'images\\' #Директория с фото для поиска объектов.
FILE = 3 #Порядковый номер файла для распознования. Отсчёт начинается с 0.
SCL_FACTOR = 1.8 #Теоретический размер искомых объектов.
NEIGHBORDS = 2 #Примерное количество объектов, расположенных рядом.

#----------------------------------------------

files = os.listdir(DIR)

image = cv2.imread(DIR + files[FILE])
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

model = cv2.CascadeClassifier(MODEL_TYPE)

objects = model.detectMultiScale(image_gray, scaleFactor = SCL_FACTOR,
                                 minNeighbors = NEIGHBORDS)

for (x, y, w, h) in objects:
    cv2.rectangle(image, (x,y,), (x + w, y + h),
                 (0,0,255), thickness = 3)

cv2.imshow("Result", image)
cv2.waitKey(0)
