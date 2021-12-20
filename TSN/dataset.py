from enum import Flag
import paddle
import os.path as path
import random
import copy
import numpy as np
from data_prepare import Compose


class FrameDataset(paddle.io.Dataset):
    def __init__(self,
                 file_path,
                 pipeline,
                 num_retries=5,
                 data_prefix=None,
                 test_mode=False,
                 suffix='img_{:05}.jpg', online=False, online_data=None):
        super(FrameDataset, self).__init__()
        self.num_retries = num_retries  # 重试的次数
        self.suffix = suffix
        self.file_path = file_path
        self.data_prefix = path.realpath(data_prefix) if \
            data_prefix is not None and path.isdir(data_prefix) else data_prefix
        self.test_mode = test_mode
        self.pipeline = pipeline
        self.online = online
        self.online_data = online_data
        self.info = self.load_file()

    def load_file(self):
        """Load index file to get video information."""
        # 从文件中加载数据信息
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                # 数据信息（帧目录-目录下存放帧的数量-标签）
                frame_dir, frames_len, labels = line_split
                if self.data_prefix is not None:
                    frame_dir = path.join(self.data_prefix, frame_dir)
                # 视频数据信息<视频目录，后缀，帧数，标签>
                info.append(dict(frame_dir=frame_dir,
                                 suffix=self.suffix,
                                 frames_len=frames_len,
                                 labels=np.int64(labels),#这里linux的话就是labels=int(labels),Windows下为labels=np.int64(labels)
                                 online=self.online))
        return info

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        # Try to catch Exception caused by reading missing frames files
        # 重试的次数
        for ir in range(self.num_retries):
            # 从数据信息中取出索引对应的视频信息，self.info 中每个元素对应的是一段视频
            results = copy.deepcopy(self.info[idx])
            try:
                # 将 <视频目录，后缀，视频帧数，视频标签> 交给 pipeline 处理
                results = self.pipeline(results)
            except Exception as e:
                print(e)
                if ir < self.num_retries - 1:
                    print("Error when loading {}, have {} trys, will try again".format(
                        results['frame_dir'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            # 返回图片集和其对应的 labels
            return results['imgs'], np.array([results['labels']])

    def prepare_test(self, idx):
        """Prepare the frames for test given index. """
        # Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            results = copy.deepcopy(self.info[idx])
            try:
                results = self.pipeline(results)
            except Exception as e:
                print(e)
                if ir < self.num_retries - 1:
                    print("Error when loading {}, have {} trys, will try again".format(
                        results['frame_dir'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            if not self.online:
                return results['imgs'], np.array([results['labels']])
            else:
                return results['imgs']

    def __len__(self):
        """get the size of the dataset."""
        return len(self.info)

    def __getitem__(self, idx):
        """ Get the sample for either training or testing given index"""
        if self.test_mode:
            return self.prepare_test(idx)
        else:
            return self.prepare_train(idx)


def online_img_convert(data):
    pipeline = Compose(online=True)
    data = pipeline(data)
    res = paddle.to_tensor(data['imgs']).unsqueeze(0)
    return res


if __name__ == '__main__':

    train_file_path = './data2/ucf101/ucf101_train_split_1_rawframes.txt'
    pipeline = Compose()
    data = FrameDataset(file_path=train_file_path,
                        pipeline=pipeline, suffix='img_{:05}.jpg')
    data_loader = paddle.io.DataLoader(
        data, num_workers=0,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        places=paddle.set_device('gpu'),
        return_list=True
    )

    for item in data_loader():
        x, y = item
        print('图片数据的 shape:', x.shape)
        print('标签数据的 shape:', y.shape)
        break
