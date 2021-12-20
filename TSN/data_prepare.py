# coding=utf-8
# 导入环境
import os
import sys
import random

import numpy as np
from paddle.fluid.data import data
import scipy.io
import cv2
from PIL import Image
import os.path as osp
import copy
from tqdm import tqdm
import time
import glob
import fnmatch
from multiprocessing import Pool, current_process

import traceback

import paddle


from collections.abc import Sequence
from collections import OrderedDict

import matplotlib.pyplot as plt
# 在notebook中使用matplotlib.pyplot绘图时，需要添加该命令进行显示


class FrameDecoder(object):
    """just parse results
    """

    def __init__(self):
        pass

    def __call__(self, results):
        # 加入数据的 format 格式信息，表示当前处理的数据类型为帧 frame
        results['format'] = 'frame'
        return results


class Sampler(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        mode(str): 'train', 'valid'
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self, num_seg, seg_len, valid_mode=False):
        self.num_seg = num_seg  # 视频分割段的数量
        self.seg_len = seg_len  # 每段中抽取帧数
        self.valid_mode = valid_mode  # train or valid

    def _get(self, frames_idx, results):
        data_format = results['format']  # 取出处理的数据类型
        # 如果处理的数据类型为帧
        if data_format == "frame":
            # 取出帧所在的目录
            frame_dir = results['frame_dir']
            imgs = []  # 存放读取到的帧图片
            for idx in frames_idx:
                # 读取图片
                img = Image.open(os.path.join(
                    frame_dir, results['suffix'].format(idx))).convert('RGB')
                # 将读取到的图片存放到列表中
                imgs.append(img)
        else:
            raise NotImplementedError
        results['imgs'] = imgs  # 添加 imgs 信息
        return results

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = results['frames_len']  # 视频中总的帧数
        average_dur = int(int(frames_len) / self.num_seg)  # 每段中视频的数量
        frames_idx = []  # 将采样到的索引存放到 frames_idx
        for i in range(self.num_seg):
            idx = 0  # 当前段采样的起始位置
            if not self.valid_mode:
                # 如果训练
                if average_dur >= self.seg_len:  # 如果每段中视频数大于每段中要采样的帧数
                    idx = random.randint(0, average_dur - self.seg_len)
                    # 计算在当前段内采样的起点
                    idx += i * average_dur  # i * average_dur 表示之前 i-1 段用过的帧
                elif average_dur >= 1:  # 如果每段中视频数大于 1
                    idx += i * average_dur  # 直接以当前段的起始位置作为采样的起始位置
                else:
                    idx = i  # 直接以当前段的索引作为起始位置
            else:
                # 如果测试
                if average_dur >= self.seg_len:
                    idx = (average_dur - 1) // 2  # 当前段的中间帧数
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            # 从采样位置采连续的 self.seg_len 帧
            for jj in range(idx, idx + self.seg_len):
                if results['format'] == 'frame':
                    frames_idx.append(jj + 1)  # 将采样到的帧索引加入到 frames_idx 中
                else:
                    raise NotImplementedError

        return self._get(frames_idx, results)  # 依据采样到的帧索引读取对应的图片


class OnlineSampler(object):
    def __init__(self, num_seg, seg_len):
        self.num_seg = num_seg  # 视频分割段的数量
        self.seg_len = seg_len  # 每段中抽取帧数

    def _get(self, frames_idx, results):
        # 如果处理的数据类型为帧
        res = dict()

        imgs = []  # 存放读取到的帧图片
        for idx in frames_idx:
            # 读取图片
            img = results[idx]
            # 将读取到的图片存放到列表中
            imgs.append(img)

        res['imgs'] = imgs
        return res

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = len(results)  # 视频中总的帧数
        average_dur = int(int(frames_len) / self.num_seg)  # 每段中视频的数量
        frames_idx = []  # 将采样到的索引存放到 frames_idx
        for i in range(self.num_seg):
            idx = 0  # 当前段采样的起始位置

            if average_dur >= self.seg_len:
                idx = (average_dur - 1) // 2  # 当前段的中间帧数
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
            # 从采样位置采连续的 self.seg_len 帧
            for jj in range(idx, idx + self.seg_len):
                frames_idx.append(jj + 1)  # 将采样到的帧索引加入到 frames_idx 中

        res = self._get(frames_idx, results)

        return res  # 依据采样到的帧索引读取对应的图片


class Scale(object):
    """
    Scale images.
    Args:
        short_size(float | int): Short size of an image will be scaled to the short_size.
    """

    # 将图片中短边的长度 resize 到 short_size，另一个变做相应尺度的缩放
    def __init__(self, short_size):
        self.short_size = short_size  # 短边长度

    def __call__(self, results):
        """
        Performs resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        imgs = results['imgs']  # 取出图片集
        resized_imgs = []  # 存放处理过的图片
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.size  # 当前图片的宽和高
            if (w <= h and w == self.short_size) or (h <= w and h == self.short_size):
                resized_imgs.append(img)
                continue

            if w < h:
                ow = self.short_size
                oh = int(self.short_size * 4.0 / 3.0)
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
            else:
                oh = self.short_size
                ow = int(self.short_size * 4.0 / 3.0)
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        results['imgs'] = resized_imgs  # 将处理过的图片复制给键值 imgs
        return results


class MultiScaleCrop(object):
    def __init__(
            self,
            target_size,  # NOTE: named target size now, but still pass short size in it!
            scales=None,
            max_distort=1,
            fix_crop=True,
            more_fix_crop=True):
        # resize 后的宽高
        self.target_size = target_size
        # resize 的尺度
        self.scales = scales if scales else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop

    def __call__(self, results):
        """
        Performs MultiScaleCrop operations.
        Args:
            imgs: List where wach item is a PIL.Image.
            XXX:
        results:

        """
        imgs = results['imgs']  # 取出图片集

        input_size = [self.target_size, self.target_size]

        im_size = imgs[0].size  # 取出第一张的图片的尺寸

        # get random crop offset
        def _sample_crop_size(im_size):
            # 图片的宽、图片的高
            image_w, image_h = im_size[0], im_size[1]
            # 图片宽和高中的最小值
            base_size = min(image_w, image_h)
            # 在宽和高中最小值的基础上计算多尺度的裁剪尺寸
            crop_sizes = [int(base_size * x) for x in self.scales]

            crop_h = [
                input_size[1] if abs(x - input_size[1]) < 3 else x
                for x in crop_sizes
            ]
            crop_w = [
                input_size[0] if abs(x - input_size[0]) < 3 else x
                for x in crop_sizes
            ]

            pairs = []
            for i, h in enumerate(crop_h):
                for j, w in enumerate(crop_w):
                    # |i-j| < self.max_distort
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))

            # 随机选取一个裁剪 pair
            crop_pair = random.choice(pairs)
            # 如果对裁剪 pair 进行修正
            # (w_offset,h_offset) 裁剪起始点
            if not self.fix_crop:
                w_offset = random.randint(0, image_w - crop_pair[0])
                h_offset = random.randint(0, image_h - crop_pair[1])
            else:
                w_step = (image_w - crop_pair[0]) / 4
                h_step = (image_h - crop_pair[1]) / 4

                ret = list()
                ret.append((0, 0))  # upper left
                if w_step != 0:
                    ret.append((4 * w_step, 0))  # upper right
                if h_step != 0:
                    ret.append((0, 4 * h_step))  # lower left
                if h_step != 0 and w_step != 0:
                    ret.append((4 * w_step, 4 * h_step))  # lower right
                if h_step != 0 or w_step != 0:
                    ret.append((2 * w_step, 2 * h_step))  # center

                if self.more_fix_crop:
                    ret.append((0, 2 * h_step))  # center left
                    ret.append((4 * w_step, 2 * h_step))  # center right
                    ret.append((2 * w_step, 4 * h_step))  # lower center
                    ret.append((2 * w_step, 0 * h_step))  # upper center

                    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
            w_offset, h_offset = random.choice(ret)
            # 返回裁剪的宽和高以及裁剪的起始点
            return crop_pair[0], crop_pair[1], w_offset, h_offset

        # 获取裁剪的宽和高以及裁剪的起始点
        crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
        # 对 imgs 中的每张图片做裁剪
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in imgs
        ]
        # 将裁剪的后图片 resize 到 (input_size[0], input_size[1])
        ret_img_group = [
            img.resize((input_size[0], input_size[1]), Image.BILINEAR)
            for img in crop_img_group
        ]
        # 将处理过的图片复制给键值 imgs
        results['imgs'] = ret_img_group
        return results


class RandomCrop(object):
    """
    Random crop images.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, results):
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        w, h = imgs[0].size  # 获取图片的宽和高
        th, tw = self.target_size, self.target_size  # resize 后的宽和高

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size".format(
                w, h, self.target_size)

        crop_images = []  # 存放裁剪后的图片
        # 计算随机裁剪的起始点，一段视频中对所帧裁剪的其实位置相同
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # 访问每一张图片
        for img in imgs:
            if w == tw and h == th:  # 如果原始的宽高与裁剪后的宽高相同
                crop_images.append(img)
            else:
                crop_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = crop_images  # 将处理过的图片复制给键值 imgs
        return results


class RandomFlip(object):
    """
    Random Flip images.
    Args:
        p(float): Random flip images with the probability p.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        """
        Performs random flip operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            flip_imgs: List where each item is a PIL.Image after random flip.
        """
        imgs = results['imgs']
        v = random.random()
        if v < self.p:  # 如果 v 小于 0.5
            results['imgs'] = [img.transpose(
                Image.FLIP_LEFT_RIGHT) for img in imgs]
        else:
            results['imgs'] = imgs
        return results


class CenterCrop(object):
    """
    Center crop images.
    Args:
        target_size(int): Center crop a square with the target_size from an image.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, results):
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            ccrop_imgs: List where each item is a PIL.Image after Center crop.
        """
        imgs = results['imgs']
        ccrop_imgs = []
        for img in imgs:
            w, h = img.size   # 图片的宽和高
            th, tw = self.target_size, self.target_size
            assert (w >= self.target_size) and (h >= self.target_size), \
                "image width({}) and height({}) should be larger than crop size".format(
                    w, h, self.target_size)

            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            ccrop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = ccrop_imgs
        return results


class Image2Array(object):
    """
    transfer PIL.Image to Numpy array and transpose dimensions from 'dhwc' to 'dchw'.
    Args:
        transpose: whether to transpose or not, default False. True for tsn.
    """

    def __init__(self, transpose=True):
        self.transpose = transpose

    def __call__(self, results):
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        imgs = results['imgs']
        # 将 list 转为 numpy
        np_imgs = (np.stack(imgs)).astype('float32')
        if self.transpose:
            # 对维度进行交换
            np_imgs = np_imgs.transpose(0, 3, 1, 2)  # nchw
        results['imgs'] = np_imgs  # 将处理过的图片复制给键值 imgs
        return results


class Normalization(object):
    """
    Normalization.
    Args:
        mean(Sequence[float]): mean values of different channels.
        std(Sequence[float]): std values of different channels.
        tensor_shape(list): size of mean, default [3,1,1]. For slowfast, [1,1,1,3]
    """

    def __init__(self, mean, std, tensor_shape=[3, 1, 1]):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}')
        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')
        self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
        self.std = np.array(std).reshape(tensor_shape).astype(np.float32)

    def __call__(self, results):
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """
        imgs = results['imgs']
        norm_imgs = imgs / 255.  # 除以 255
        norm_imgs -= self.mean  # 减去均值
        norm_imgs /= self.std  # 除以方差
        results['imgs'] = norm_imgs  # 将处理过的图片复制给键值 imgs
        return results


class Compose(object):
    """
    Composes several pipelines(include decode func, sample func, and transforms) together.

    Note: To deal with ```list``` type cfg temporaray, like:

        transform:
            - Crop: # A list
                attribute: 10
            - Resize: # A list
                attribute: 20

    every key of list will pass as the key name to build a module.
    XXX: will be improved in the future.

    Args:
        pipelines (list): List of transforms to compose.
    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.
    """

    def __init__(self, train_mode=False, online=False):
        # assert isinstance(pipelines, Sequence)
        self.pipelines = list()
        if not online:
            self.pipelines.append(FrameDecoder())
            if train_mode:
                self.pipelines.append(
                    Sampler(num_seg=8, seg_len=1, valid_mode=False))
            else:
                self.pipelines.append(
                    Sampler(num_seg=8, seg_len=1, valid_mode=True))
        else:
            self.pipelines.append(
                OnlineSampler(num_seg=8, seg_len=1)
            )
        self.pipelines.append(Scale(short_size=256))
        if train_mode:
            self.pipelines.append(MultiScaleCrop(target_size=256))
            self.pipelines.append(RandomCrop(target_size=224))
            self.pipelines.append(RandomFlip())
        else:
            self.pipelines.append(CenterCrop(target_size=224))
        self.pipelines.append(Image2Array())
        self.pipelines.append(Normalization(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    def __call__(self, data):
        # 将传入的 data 依次经过 pipelines 中对象处理
        for p in self.pipelines:
            try:
                data = p(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(p, e, str(stack_info)))
                raise e
        return data
