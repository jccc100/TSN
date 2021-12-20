import paddle.nn as nn
import paddle.nn.initializer as init
import os.path as osp
import paddle
from tqdm import tqdm
import time
from paddle.io import Dataset
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, BatchNorm, AdaptiveAvgPool2D, AvgPool2D, BatchNorm2D

from paddle import ParamAttr
from paddle.regularizer import L2Decay


import paddle.nn.functional as F
import paddle.nn as nn
import math


def weight_init_(layer,
                 func,
                 weight_name=None,
                 bias_name=None,
                 bias_value=0.0,
                 **kwargs):
    """
    In-place params init function.
    Usage:
    .. code-block:: python

        import paddle
        import numpy as np

        data = np.ones([3, 4], dtype='float32')
        linear = paddle.nn.Linear(4, 4)
        input = paddle.to_tensor(data)
        print(linear.weight)
        linear(input)

        weight_init_(linear, 'Normal', 'fc_w0', 'fc_b0', std=0.01, mean=0.1)
        print(linear.weight)
    """

    if hasattr(layer, 'weight') and layer.weight is not None:
        getattr(init, func)(**kwargs)(layer.weight)
        if weight_name is not None:
            # override weight name
            layer.weight.name = weight_name

    if hasattr(layer, 'bias') and layer.bias is not None:
        init.Constant(bias_value)(layer.bias)
        if bias_name is not None:
            # override bias name
            layer.bias.name = bias_name


def load_ckpt(model, weight_path):
    """
    """
    # model.set_state_dict(state_dict)

    if not osp.isfile(weight_path):
        raise IOError(f'{weight_path} is not a checkpoint file')
    # state_dicts = load(weight_path)

    state_dicts = paddle.load(weight_path)
    tmp = {}
    total_len = len(model.state_dict())
    with tqdm(total=total_len,
              position=1,
              bar_format='{desc}',
              desc="Loading weights") as desc:
        for item in tqdm(model.state_dict(), total=total_len, position=0):
            name = item
            desc.set_description('Loading %s' % name)
            tmp[name] = state_dicts[name]
            time.sleep(0.01)
        ret_str = "loading {:<20d} weights completed.".format(
            len(model.state_dict()))
        desc.set_description(ret_str)
        model.set_state_dict(tmp)


class ConvBNLayer(nn.Layer):
    """Conv2D and BatchNorm2D layer.
    Args:
        in_channels (int): Number of channels for the input.
        out_channels (int): Number of channels for the output.
        kernel_size (int): Kernel size.
        stride (int): Stride in the Conv2D layer. Default: 1.
        groups (int): Groups in the Conv2D, Default: 1.
        act (str): Indicate activation after BatchNorm2D layer.
        name (str): the name of an instance of ConvBNLayer.
    Note: weight and bias initialization include initialize values and name the restored parameters, values initialization are explicit declared in the ```init_weights``` method.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2,
                            groups=groups,
                            weight_attr=ParamAttr(),
                            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._act = act

        self._batch_norm = BatchNorm2D(out_channels,
                                       weight_attr=ParamAttr(),
                                       bias_attr=ParamAttr())

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._act:
            y = getattr(paddle.nn.functional, self._act)(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 act="relu",
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 act="relu",
                                 name=name + "_branch2b")

        self.conv2 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels * 4,
                                 kernel_size=1,
                                 act=None,
                                 name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels * 4,
                                     kernel_size=1,
                                     stride=stride,
                                     name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        return F.relu(y)


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 filter_size=3,
                                 stride=stride,
                                 act="relu",
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 filter_size=3,
                                 act=None,
                                 name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     filter_size=1,
                                     stride=stride,
                                     name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = F.relu(y)
        return y


class ResNet(nn.Layer):
    """ResNet backbone.
    Args:
        depth (int): Depth of resnet model.
        pretrained (str): pretrained model. Default: None.
    """

    def __init__(self, depth, pretrained=None, name='conv1'):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.layers = depth
        self.name = name

        supported_layers = [18, 34, 50, 101, 152]
        assert self.layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, self.layers)

        if self.layers == 18:
            depth = [2, 2, 2, 2]
        elif self.layers == 34 or self.layers == 50:
            depth = [3, 4, 6, 3]
        elif self.layers == 101:
            depth = [3, 4, 23, 3]
        elif self.layers == 152:
            depth = [3, 8, 36, 3]

        in_channels = [64, 256, 512, 1024]
        out_channels = [64, 128, 256, 512]

        self.conv = ConvBNLayer(in_channels=3,
                                out_channels=64,
                                kernel_size=7,
                                stride=2,
                                act="relu",
                                name=self.name)
        self.pool2D_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if self.layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if self.layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        conv_name,
                        BottleneckBlock(
                            in_channels=in_channels[block]
                            if i == 0 else out_channels[block] * 4,
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            name=conv_name))

                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(in_channels=in_channels[block]
                                   if i == 0 else out_channels[block],
                                   out_channels=out_channels[block],
                                   stride=2 if i == 0 and block != 0 else 1,
                                   shortcut=shortcut,
                                   name=conv_name))
                    self.block_list.append(basic_block)
                    shortcut = True

    def init_weights(self):
        """Initiate the parameters.
        Note:
            1. when indicate pretrained loading path, will load it to initiate backbone.
            2. when not indicating pretrained loading path, will follow specific initialization initiate backbone. Always, Conv2D layer will be initiated by KaimingNormal function, and BatchNorm2d will be initiated by Constant function.
            Please refer to https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/initializer/kaiming/KaimingNormal_en.html
        """
        # XXX: check bias!!! check pretrained!!!
        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    # XXX: no bias
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, inputs):
        """Define how the backbone is going to run.
        """
        # NOTE: Already merge axis 0(batches) and axis 1(channels) before extracting feature phase,
        # please refer to paddlevideo/modeling/framework/recognizers/recognizer2d.py#L27
        # y = paddle.reshape(
        #    inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])

        y = self.conv(inputs)
        y = self.pool2D_max(y)
        for block in self.block_list:
            y = block(y)
        return y


class TSNHead(nn.Layer):
    """TSN Head.
    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        drop_ratio(float): drop ratio. Default: 0.4.
        std(float): Std(Scale) value in normal initilizar. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to initialize.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 drop_ratio=0.4,
                 ls_eps=0.,
                 std=0.01,
                 **kwargs):

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels  # 分类层输入的通道数
        self.drop_ratio = drop_ratio  # dropout 比例
        self.stdv = 1.0 / math.sqrt(self.in_channels * 1.0)
        self.std = std

        # NOTE: global pool performance
        self.avgpool2d = AdaptiveAvgPool2D((1, 1))

        if self.drop_ratio != 0:
            self.dropout = Dropout(p=self.drop_ratio)
        else:
            self.dropout = None

        self.fc = Linear(self.in_channels, self.num_classes)

        self.loss_func = paddle.nn.CrossEntropyLoss()  # 损失函数
        self.ls_eps = ls_eps  # 标签平滑系数

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc,
                     'Normal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.,
                     std=self.std)

    def forward(self, x, seg_num):
        """Define how the head is going to run.
        Args:
            x (paddle.Tensor): The input data.
            num_segs (int): Number of segments.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """

        # XXX: check dropout location!
        # [N * num_segs, in_channels, 7, 7]

        x = self.avgpool2d(x)
        # [N * num_segs, in_channels, 1, 1]
        x = paddle.reshape(x, [-1, seg_num, x.shape[1]])
        # [N, seg_num, in_channels]
        x = paddle.mean(x, axis=1)
        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels]
        score = self.fc(x)
        # [N, num_class]
        # x = F.softmax(x)  #NOTE remove
        return score

    def loss(self, scores, labels, reduce_sum=False, **kwargs):
        """Calculate the loss accroding to the model output ```scores```,
           and the target ```labels```.

        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.

        Returns:
            losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).

        """
        if len(labels) == 1:
            labels = labels[0]
        else:
            raise NotImplemented

        # 如果标签平滑系数不等于 0
        if self.ls_eps != 0.:
            labels = F.one_hot(labels, self.num_classes)
            labels = F.label_smooth(labels, epsilon=self.ls_eps)
            # reshape [bs, 1, num_classes] to [bs, num_classes]
            # NOTE: maybe squeeze is helpful for understanding.
            labels = paddle.reshape(labels, shape=[-1, self.num_classes])
        # labels.stop_gradient = True  #XXX(shipping): check necessary
        losses = dict()
        # NOTE(shipping): F.crossentropy include logsoftmax and nllloss !
        # NOTE(shipping): check the performance of F.crossentropy
        loss = self.loss_func(scores, labels, **kwargs)  # 计算损失
        avg_loss = paddle.mean(loss)
        top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
        top3 = paddle.metric.accuracy(input=scores, label=labels, k=3)

        # _, world_size = get_dist_info()
        #
        # # NOTE(shipping): deal with multi cards validate
        # if world_size > 1 and reduce_sum:
        #     top1 = paddle.distributed.all_reduce(top1, op=paddle.distributed.ReduceOp.SUM) / world_size
        #     top5 = paddle.distributed.all_reduce(top5, op=paddle.distributed.ReduceOp.SUM) / world_size

        losses['top1'] = top1
        losses['top5'] = top3
        losses['loss'] = avg_loss

        return losses


class Recognizer2D(paddle.nn.Layer):
    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.backbone.init_weights()
        self.head = head
        self.head.init_weights()

    def extract_feature(self, imgs):
        """Extract features through a backbone.

        Args:
        imgs (paddle.Tensor) : The input images.

        Returns:
            feature (paddle.Tensor) : The extracted features.
        """
        feature = self.backbone(imgs)
        return feature

    def forward(self, imgs, **kwargs):
        """Define how the model is going to run, from input to output.
        """
        batches = imgs.shape[0]  # 批次大小
        num_segs = imgs.shape[1]  # 分割的帧数
        # 对 imgs 进行 reshape，[N,T,C,H,W]->[N*T,C,H,W]
        imgs = paddle.reshape(imgs, [-1] + list(imgs.shape[2:]))
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature, num_segs)
        return cls_score

    """2D recognizer model framework."""

    def train_step(self, data_batch, reduce_sum=False):
        """Define how the model is going to train, from input to output.
        """
        # NOTE: As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase,
        # should obtain it from imgs(paddle.Tensor) now, then call self.head method.

        # labels = labels.squeeze()
        # XXX: unsqueeze label to [label] ?

        imgs = data_batch[0]  # 从批次中取出训练数据
        labels = data_batch[1:]  # 从批次中取出数据对应的标签
        cls_score = self(imgs)  # 计算预测分数
        loss_metrics = self.head.loss(cls_score, labels, reduce_sum)  # 计算损失
        return loss_metrics

    def val_step(self, data_batch, reduce_sum=True):
        return self.train_step(data_batch, reduce_sum=reduce_sum)

    def test_step(self, data_batch, reduce_sum=False):
        """Define how the model is going to test, from input to output."""
        # NOTE: (shipping) when testing, the net won't call head.loss,
        # we deal with the test processing in /paddlevideo/metrics
        imgs = data_batch[0]  # 从批次中取出训练数据
        cls_score = self(imgs)  # 计算预测分数
        return cls_score
