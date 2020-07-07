import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
model = load_model("models/cnnCat.h5")

image_path = list(paths.list_images("TestSet/"))
for imagePath in image_path:
    imageLoad = cv2.imread(imagePath, 0)
    imageLoad = cv2.resize(imageLoad, (24, 24))
    print(imageLoad.shape)
    image = img_to_array(imageLoad)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    predict = model.predict_classes(image)
    result = ""
    if predict[0] == 0:
        result = "Closed"
    if predict[0] == 1 :
        result = "Open"
    print(imagePath + "     " + result + " " + str(predict[0]))

    cv2.imshow("image", imageLoad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#
# imageLoad = cv2.imread("l_eye_closed.jpg", cv2.COLOR_BGR2BGRA)
# imageLoad = cv2.resize(imageLoad, (24,24))
# image_array = img_to_array(imageLoad)
# image_array = np.expand_dims(image_array, axis=0)
# print(image_array.shape)
# predict = model.predict_classes(image_array)
# print(predict[0])
# cv2.imshow("image",imageLoad)
# cv2.waitKey(0)
# cv2.destroyAllWindows()