import copy
import os
from collections import OrderedDict

import numpy as np
import paddle
from tsn import ResNet, TSNHead, Recognizer2D
from data_prepare import Compose
from dataset import FrameDataset
import time
import os.path as osp

from config import *

from paddle.distributed import fleet

Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('COLORING', True):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message


def build_record(framework_type):
    record_list = [
        ("loss", AverageMeter('loss', '7.5f')),
        ("lr", AverageMeter('lr', 'f', need_avg=False)),
        ("batch_time", AverageMeter('elapse', '.3f')),
        ("reader_time", AverageMeter('reader', '.3f')),
    ]

    if 'Recognizer' in framework_type:
        record_list.append(("top1", AverageMeter("top1", '.5f')))
        record_list.append(("top5", AverageMeter("top5", '.5f')))

    record_list = OrderedDict(record_list)
    return record_list


def log_batch(metric_list, batch_id, epoch_id, total_epoch, mode, ips):
    metric_str = ' '.join([str(m.value) for m in metric_list.values()])
    epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch_id, total_epoch)
    step_str = "{:s} step:{:<4d}".format(mode, batch_id)
    print("{:s} {:s} {:s}s {}".format(
        coloring(epoch_str, "HEADER") if batch_id == 0 else epoch_str,
        coloring(step_str, "PURPLE"), coloring(metric_str, 'OKGREEN'), ips))


def log_epoch(metric_list, epoch, mode, ips):
    metric_avg = ' '.join([str(m.mean) for m in metric_list.values()] +
                          [metric_list['batch_time'].total])

    end_epoch_str = "END epoch:{:<3d}".format(epoch)

    print("{:s} {:s} {:s}s {}".format(coloring(end_epoch_str, "RED"),
                                      coloring(mode, "PURPLE"),
                                      coloring(metric_avg, "OKGREEN"),
                                      ips))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name='', fmt='f', need_avg=True):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        if isinstance(val, paddle.Tensor):
            val = val.numpy()[0]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(s=self.sum / 60,
                                                            self=self)

    @property
    def mean(self):
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)


def train_model(validate=True):
    # 模型输出目录
    output_dir = f"./output/{model_name}"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            pass

    # 初始化训练参数
    fleet.init(is_collective=True)

    # 1. Construct model （创建模型）
    tsn = ResNet(depth=depth, pretrained=pretrained)
    head = TSNHead(num_classes=num_classes, in_channels=in_channels)
    model = Recognizer2D(backbone=tsn, head=head)

    # 2. Construct dataset and dataloader
    train_pipeline = Compose(train_mode=True)
    train_dataset = FrameDataset(#这里要注意Linux格式与Windows格式不兼容问题，见https://blog.csdn.net/qq_15821487/article/details/114835293
        file_path=train_file_path, pipeline=train_pipeline, suffix=suffix)

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=drop_last
    )

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        places=paddle.set_device('gpu'),
        num_workers=num_workers,
        return_list=return_list
    )

    if validate:
        valid_pipeline = Compose(train_mode=False)
        valid_dataset = FrameDataset(
            file_path=valid_file_path, pipeline=valid_pipeline, suffix=suffix)
        valid_sampler = paddle.io.DistributedBatchSampler(
            valid_dataset,
            batch_size=batch_size,
            shuffle=valid_shuffle,
            drop_last=drop_last
        )
        valid_loader = paddle.io.DataLoader(
            valid_dataset,
            batch_sampler=valid_sampler,
            places=paddle.set_device('gpu'),
            num_workers=num_workers,
            return_list=return_list
        )

    # 3. Construct solver.
    # 学习率的衰减策略
    lr = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries, values=values)
    # 使用的优化器
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        momentum=momentum,
        parameters=model.parameters(),
        weight_decay=paddle.regularizer.L2Decay(weight_decay)
    )

    optimizer = fleet.distributed_optimizer(optimizer)

    model = fleet.distributed_model(model)

    # 4. Train Model
    best = 0.
    for epoch in range(0, epochs):
        model.train()  # 将模型设置为训练模式
        record_list = build_record(framework)
        tic = time.time()
        # 访问每一个 batch
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)
            # 4.1 forward
            #windows下的int数据类型和linux下的int数据类型不一致，详见这个https://blog.csdn.net/qq_15821487/article/details/114835293
            #要到dataset.py的45行中去调整
            outputs = model.train_step(data)  # 执行前向推断
            # 4.2 backward
            # 反向传播
            avg_loss = outputs['loss']
            avg_loss.backward()
            # 4.3 minimize
            # 梯度更新
            optimizer.step()
            optimizer.clear_grad()

            # log record
            record_list['lr'].update(
                optimizer._global_learning_rate(), batch_size)
            for name, value in outputs.items():
                record_list[name].update(value, batch_size)

            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % log_interval == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, epochs, "train", ips)

        # learning rate epoch step
        lr.step()

        ips = "ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum
        )
        log_epoch(record_list, epoch + 1, "train", ips)

        def evaluate(best):
            model.eval()
            record_list = build_record(framework)
            record_list.pop('lr')
            tic = time.time()
            for i, data in enumerate(valid_loader):
                outputs = model.val_step(data)

                # log_record
                for name, value in outputs.items():
                    record_list[name].update(value, batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % log_interval == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, epochs, "val", ips)

            ips = "ips: {:.5f} instance/sec.".format(
                batch_size *
                record_list["batch_time"].count / record_list["batch_time"].sum
            )
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            for top_flag in ['hit_at_one', 'top1']:
                if record_list.get(top_flag) and record_list[top_flag].avg > best:
                    best = record_list[top_flag].avg
                    best_flag = True

            return best, best_flag

         # 5. Validation
        if validate or epoch == epochs - 1:
            with paddle.fluid.dygraph.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                paddle.save(optimizer.state_dict(), osp.join(
                    output_dir, model_name + "_best.pdopt"))
                paddle.save(model.state_dict(), osp.join(
                    output_dir, model_name + "_best.pdparams"))
                if model_name == "AttentionLstm":
                    print(f"Already save the best model (hit_at_one){best}")
                else:
                    print(
                        f"Already save the best model (top1 acc){int(best * 10000) / 10000}")

        # 6. Save model and optimizer
        if epoch % save_interval == 0 or epoch == epochs - 1:
            paddle.save(optimizer.state_dict(), osp.join(
                output_dir, model_name + f"_epoch_{epoch + 1:05d}.pdopt"))
            paddle.save(model.state_dict(), osp.join(
                output_dir, model_name + f"_epoch_{epoch + 1:05d}.pdparams"))

    print(f'training {model_name} finished')


# 在执行代码过程中，如果出现 ‘ValueError: parameter name [conv1_weights] have be been used’ 问题，
# 可以点击上方的第三个按钮 ‘重启并运行全部’ 来解决
train_model(True)  # 训练模型时取消注释
