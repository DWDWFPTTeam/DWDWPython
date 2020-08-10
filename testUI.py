import cv2
from PIL import Image
from PIL import ImageTk
import threading
import tkinter as tk
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from pygame import mixer
import time, sys
import requests
import threading

def startClicked(videoloop_stop):
    threading.Thread(target=videoLoop, args=(videoloop_stop,)).start()


def stopClicked(videoloop_stop):
    videoloop_stop[0] = True

def sendRecord(path, frame):
    cv2.imwrite(os.path.join(path, 'sleep.jpg'), frame)
    files = {'image': open('sleep.jpg', 'rb')}
    r = requests.post('https://dwdw-api-on0.conveyor.cloud/api/Record/SaveRecord', data={
        "deviceCode": "D001",
        "type": 3
    }, files=files)
    print(r)

def videoLoop(mirror=False):
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')
    model = load_model('models/cnnCat.h5')
    cap = cv2.VideoCapture(0)
    score = 0
    rpred = [99]
    path = os.getcwd()
    requestFlag = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    while True:
        ret, frame = cap.read()
        if mirror is True:
            frame = frame[:, ::-1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        right_eye = reye.detectMultiScale(gray)

        if len(right_eye) == 0:
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
            if rpred[0] == 0:
                score += 1
                requestFlag += 1
            else:
                score -= 1

        if score < 0:
            score = 0

        if score > 5:
            score = 5

        if score == 0:
            requestFlag = 0

        if score == 5:
            try:
                sound.play()
                if requestFlag == 5:
                    t1 = threading.Thread(target=sendRecord(path, frame))
                    t1.start()
            except:
                isplaying = False
                pass

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image)
        panel = tk.Label(image=image)
        panel.image = image
        panel.place(x=50, y=50)

        # check switcher value
        if videoloop_stop[0]:
            # if switcher tells to stop then we switch it again and stop videoloop
            videoloop_stop[0] = False
            panel.destroy()
            break


# videoloop_stop is a simple switcher between ON and OFF modes
videoloop_stop = [False]

root = tk.Tk()
root.geometry("900x600+0+0")

button1 = tk.Button(
    root, text="start", bg="#fff", font=("", 20),
    command=lambda: startClicked(videoloop_stop))
button1.place(x=700, y=50, width=150, height=100)

button2 = tk.Button(
    root, text="stop", bg="#fff", font=("", 20),
    command=lambda: stopClicked(videoloop_stop))
button2.place(x=700, y=200, width=150, height=100)

root.mainloop()