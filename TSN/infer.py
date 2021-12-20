import paddle
from tsn import ResNet, TSNHead, Recognizer2D
import paddle.nn.functional as F
from PIL import Image
import cv2
import imageio
from dataset import online_img_convert
from config import *


index_class = [x.strip().split() for x in open(
    './data2/ucf101/annotations/classInd.txt')]

# index_class = [x.strip().split() for x in open(
#     './data2/ucf101/annotations/classInd.txt')]
@paddle.no_grad()
def inference(data):
    model_file = './output/MainTSN/MainTSN_best.pdparams'#选定模型
    # 1. Construct dataset and dataloader.

    # 1. Construct model.
    # 创建模型
    tsn = ResNet(depth=depth, pretrained=None)
    head = TSNHead(num_classes=num_classes, in_channels=in_channels)
    model = Recognizer2D(backbone=tsn, head=head)
    # 将模型设置为评估模式
    model.eval()
    # 加载权重
    state_dicts = paddle.load(model_file)
    model.set_state_dict(state_dicts)

    outputs = model.test_step(data)
    scores = F.softmax(outputs)
    class_id = paddle.argmax(scores, axis=-1)
    pred = class_id.numpy()[0]

    return index_class[pred][1]


# 启动推理
# 在执行代码过程中，如果出现 ‘ValueError: parameter name [conv1_weights] have be been used’ 问题，
# 可以点击上方的第三个按钮 ‘重启并运行全部’ 来解决
# inference()  # 模型推理时取消注释
# video = imageio.get_reader('./data2/ucf101/videos/Fencing/v_Fencing_g01_c02.avi', 'ffmpeg')#选定要测试的视频
video = imageio.get_reader('./data2/ucf101/videos/Execute/A35-3-6_(new).avi', 'ffmpeg')#选定要测试的视频
frame_list = []
for frame in video.iter_data():
    frame_list.append(Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
data = online_img_convert(frame_list)
data = [data]
res = inference(data)
print(res)



