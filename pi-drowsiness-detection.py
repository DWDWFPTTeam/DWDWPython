import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from pygame import mixer
import time, sys
import requests

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnnCat.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
count = 0
score = 0
thicc = 5
rpred = [99]
lpred = [99]
deviceCode = "D001"
requestFlag = 0

while(True):

    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    if len(left_eye) == 0 and len(right_eye) == 0:
        score += 1
        requestFlag += 1
    else:
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
            score += 1
            requestFlag += 1
        else:
            score -= 1

    if score == 0:
        requestFlag = 0

    if score < 0:
        score = 0

    if score == 5:
        try:
             if requestFlag == 5:
                 r = requests.get('https://dwdw-api-on0.conveyor.cloud/Test/TestNotify?room=1')
                 print(r)
             sound.play()
             cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        except:
            isplaying = False
            pass

    if score > 5:
        score = 5

 
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
root.mainloop()

