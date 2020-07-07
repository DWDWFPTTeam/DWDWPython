import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# from pygame import mixer
import time, sys

# mixer.init()
# sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier(
    "/Users/admin/PycharmProjects/DrowsinessDetectionProjectPi/haar cascade files/haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier(
    "/Users/admin/PycharmProjects/DrowsinessDetectionProjectPi/haar cascade files/haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier(
    "/Users/admin/PycharmProjects/DrowsinessDetectionProjectPi/haar cascade files/haarcascade_righteye_2splits.xml")

lbl = ['Close', 'Open']

model = load_model('/Users/admin/PycharmProjects/DrowsinessDetectionProjectPi/models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
count = 0
score = 0
thicc = 5
rpred = [99]
lpred = [99]
deviceId = 1
# out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (800, 600))
while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)


    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = img_to_array(r_eye)
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, 1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = img_to_array(l_eye)
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, 1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score = score + 1
    else:
        score = score - 1

    if score < 0:
        score = 0

    if score > 6:
        score = 6

    if score >= 5:
        # out.write(frame)
        # cv2.imwrite(os.path.join(path, 'sleep.jpg'), frame)
        os.system('say "Do not sleep"')
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
