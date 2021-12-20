import paddle
from data_prepare import Compose
import numpy as np
from dataset import FrameDataset
from tsn import ResNet, TSNHead, Recognizer2D

from config import *


class CenterCropMetric(object):
    def __init__(self, data_size, batch_size, log_interval=20):
        """prepare for metrics
        """
        super().__init__()
        self.data_size = data_size
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.top1 = []
        self.top5 = []

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        labels = data[1]

        top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        top3 = paddle.metric.accuracy(input=outputs, label=labels, k=3)

        self.top1.append(top1.numpy())
        self.top5.append(top3.numpy())
        # preds ensemble
        if batch_id % self.log_interval == 0:
            print("[TEST] Processing batch {}/{} ...".format(batch_id,
                  self.data_size // self.batch_size))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        print('[TEST] finished, avg_acc1= {}, avg_acc5= {} '.format(
            np.mean(np.array(self.top1)), np.mean(np.array(self.top5)))
        )


@paddle.no_grad()
def test_model(weights):
    # 1. Construct dataset and dataloader.
    test_pipeline = Compose(train_mode=False)
    test_dataset = FrameDataset(
        file_path=valid_file_path, pipeline=test_pipeline, suffix=suffix)
    test_sampler = paddle.io.DistributedBatchSampler(
        test_dataset,
        batch_size=batch_size,
        shuffle=valid_shuffle,
        drop_last=drop_last
    )
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        places=paddle.set_device('gpu'),
        num_workers=num_workers,
        return_list=return_list
    )

    # 1. Construct model.
    # 创建模型
    tsn_test = ResNet(depth=depth, pretrained=None)  # ,name='conv1_test'
    head = TSNHead(num_classes=num_classes, in_channels=in_channels)
    model = Recognizer2D(backbone=tsn_test, head=head)
    # 将模型设置为评估模式
    model.eval()
    # 加载权重
    state_dicts = paddle.load(weights)
    model.set_state_dict(state_dicts)

    # add params to metrics
    data_size = len(test_dataset)

    metric = CenterCropMetric(data_size=data_size, batch_size=batch_size)
    for batch_id, data in enumerate(test_loader):
        # 预测
        outputs = model.test_step(data)
        metric.update(batch_id, data, outputs)
    metric.accumulate()


model_file = './output/MainTSN/MainTSN_best.pdparams'
# 在执行代码过程中，如果出现 ‘ValueError: parameter name [conv1_weights] have be been used’ 问题，
# 可以点击上方的第三个按钮 ‘重启并运行全部’ 来解决
test_model(model_file) # 模型评估时取消注释
