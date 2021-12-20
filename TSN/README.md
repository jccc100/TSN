# TSN 视频分类模型
当前接口为
```url
http:ip:port/getclass_tsn', 
```
接受post请求，请求体带有video标签，并附带有一个视频文件。
接口返回
```json
{'label':'视频类别分类结果'}
```
上述接口定义可在main.py中进行修改

## 视频分类训练
训练需根据数据集来修改extract_frames.py和config.py中对应的路径，使其与对应数据集相对应，
extrace_frames.py负责对视频的帧分割工作
需修改perpare_dir中的一些逻辑生成对应的分类和文件路径映射文件
目前数据集采用UCF101数据集进行

## 
