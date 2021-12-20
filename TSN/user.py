import base64
from flask import jsonify
import requests
import json

# count = 1

url = 'http://localhost:5003/getclass_tsn'

filename='./data2/ucf101/videos/testVideo/202111221947_(new).avi'
# files = {"files": open(filename, 'rb'), "filename": filename}
data = {'filename':filename}
r = requests.post(url=url,json=data)
print(r.text)

#tips:尝试用Django可以上传图片