# TSN
#文件说明
config.py:配置文件  训练数据位置需要根据实际情况修改\
data_prepare.py：生成数据标签\
extract_fames.py：将视频提取成帧图片保存\
main.py：web接口\
train.py：训练\
eval.py：验证\
infer.py：推理\
data2:存放数据，将百度云里面的data2替换即可\
data2/ucf101/annotations:动作类别、训练标签、测试标签\
data2/ucf101/rawframes：视频分帧以后的图片存放位置\
data2/ucf101/videos:视频文件存放位置\
output:训练参数文件，将百度云中的output文件夹拿过来替换即可
#训练
1.修改config.py中的配置\
2.运行train.py文件
#推理
运行infer.py,在最后指定要推理的视频文件路径
#其他说明
①现在项目是在windows环境下训练的，换成Linux中训练需要修改config.py中部分路径与训练标签文件中的路径（data2/ucf101/ucf101_train_split_1_rawframes.txt等）\
②aistudio开源代码参考：https://aistudio.baidu.com/aistudio/projectdetail/2250682?channelType=0&channel=0 \
③数据以及预训练文件：链接：https://pan.baidu.com/s/1DW9-1opc_Nc7Cl3b9HGEAg 提取码：uivk  （里面是整个项目包括数据预训练文件）

