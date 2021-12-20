from PIL import Image
from flask import Flask, request
import cv2
import imageio
from dataset import online_img_convert
from infer import inference

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    return "<h1>Hello World</h1>"


@app.route('/getclass_tsn', methods=['POST'])
def get_class_use_tsn():
    myjson = request.get_json()
    file = myjson.get('filename')
    print(file)
    video = imageio.get_reader(file, 'ffmpeg')
    frame_list = []
    for frame in video.iter_data():
        frame_list.append(Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
    data = online_img_convert(frame_list)
    data = [data]
    res = inference(data)
    return {
        'video_label': res
    }, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5003)
