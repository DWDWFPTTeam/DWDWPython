import cv2
import os
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from pygame import mixer
import time, sys
import requests
import threading
import boto3
import time
  


key = "image"
bucket = 'datndd-drowsy-bucket'
path = os.getcwd()


mixer.init()
sound = mixer.Sound('alarm.wav')

reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnnCat.h5')

cap = cv2.VideoCapture(0)
count = 0
score = 0
rpred = [99]
deviceCode = "D001"
requestFlag = 0

def sendRecord():
    cv2.imwrite(os.path.join(path, 'sleep.jpg'), frame)
    print("send_record to S3")
    client = boto3.client('s3', region_name='us-east-1')
    # ts stores the time in seconds
    ts = time.time()
    # print the current timestamp
    file_name = f"sleep_{ts}.jpg"
    client.upload_file('./sleep.jpg', bucket, file_name)
    
    # files = {'image': open('sleep.jpg', 'rb')}
    # r = requests.post('https://dwdw-api-on0.conveyor.cloud/api/Record/SaveRecord', data={
    #     "deviceCode": "D001",
    #     "type": 3
    # }, files=files)
    print(f"success send {file_name} to S3!!")
    print(ts)


while(True):

    ret, frame = cap.read()
    height, width = frame.shape[:2]
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
            # rpred = model.predict_classes(r_eye)
            predict_x=model.predict(r_eye) 
            rpred=np.argmax(predict_x,axis=1)
            break
        if rpred[0] == 0:
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
            sound.play()
            if requestFlag == 5:
                threading.Thread(target=sendRecord, args=()).start()
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

