import cv2
from imutils import paths
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import random
import numpy as np
import os

face = cv2.CascadeClassifier("/Users/admin/PycharmProjects/Drowsiness detection/haar cascade files/haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("/Users/admin/PycharmProjects/Drowsiness detection/haar cascade files/haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("/Users/admin/PycharmProjects/Drowsiness detection/haar cascade files/haarcascade_righteye_2splits.xml")



image_paths = list(paths.list_images("dataset_B_FacialImages/OpenFace"))
for (j, imagePath) in enumerate(image_paths):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    for (x, y, w, h) in right_eye:
        r_eye = image[y:y + h, x:x + w]
        cv2.imwrite("Dataset3/1/openRightEyes/closedRightEye" + str(j) + ".jpg", r_eye)
    for (x, y, w, h) in left_eye:
        l_eye = image[y:y + h, x:x + w]
        cv2.imwrite("Dataset3/1/openLeftEyes/closedLeftEye" + str(j) + ".jpg", l_eye)

