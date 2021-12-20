# data
suffix = 'img_{:05}.jpg'    # 图片后缀名
batch_size = 2             # 批次大小
num_workers = 0             # 使用 work 
drop_last = True            
return_list = True

## train data
train_file_path = './data2/ucf101/ucf101_train_split_1_rawframes.txt'  # 训练数据
train_shuffle = True   # 是否进行混淆操作

## valid data
valid_file_path = './data2/ucf101/ucf101_val_split_1_rawframes.txt'    # 验证数据
valid_shuffle = False  # 是否进行混淆操作

# model
framework = 'Recognizer2D'
# model_name = 'TSN'   # 模型名1
model_name = 'MainTSN'   # 模型名2
depth = 50           # ResNet 网络深度
#下面这个一定要注意改
num_classes = 3    # 类别数
in_channels = 2048   # 最后一层 channel 数
drop_ratio = 0.5     # dropout 比例
pretrained = None    # 预训练模型参数文件

# lr
boundaries = [40, 60]            # 学习率更新的轮
values = [0.01, 0.001, 0.0001]   # 学习率修改对映的值

# optimizer
momentum = 0.9         # 动量更新系数
weight_decay = 1e-4    # 权重衰减系数

# train
log_interval = 20    # 每隔多少步打印一次信息
save_interval = 10   # 每隔多少轮保存一次模型参数
epochs = 80          # 总共训练的轮数
log_level = 'INFO' 
