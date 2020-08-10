import requests

files = {'image': open('l_eye_closed.jpg', 'rb')}
headers = {"Content-Type":"multipart/form-data"}

r = requests.post('https://dwdw-api-on0.conveyor.cloud/api/Record/SaveRecord', data={
      "deviceCode": "D001",
      "type": 3
    }, files = files)
print(r)
