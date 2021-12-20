import cv2
import os
import sys
import glob
from multiprocessing import Pool


# 提取视频文件的frames


out_dir=r'./data2/ucf101/rawframes/WithGun'
src_dir = r'./data2/ucf101/videos/WithGun'
ext = 'avi'
num_worker = 8

level = 1


def dump_frames(vid_item):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = os.path.join(out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    vr = cv2.VideoCapture(full_path)
    videolen = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(videolen):
        ret, frame = vr.read()
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        # covert the BGR img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is not None:
            # cv2.imwrite will write BGR into RGB images
            cv2.imwrite('{}/img_{:05d}.jpg'.format(out_full_path, i + 1), img)
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, videolen))
            break
    print('{} done with {} frames'.format(vid_name, videolen))
    sys.stdout.flush()
    return True

# 多进程的方式提取视频帧


def extract_frames():
    if not os.path.isdir(out_dir):
        print('Creating folder: {}'.format(out_dir))
        os.makedirs(out_dir)
    if level == 2:
        classes = os.listdir(src_dir)
        for classname in classes:
            new_dir = os.path.join(out_dir, classname)
            if not os.path.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', src_dir)
    print('Extension of videos: ', ext)
    if level == 2:
        fullpath_list = glob.glob(src_dir + '/*/*.' + ext)
        done_fullpath_list = glob.glob(out_dir + '/*/*')
    elif level == 1:
        fullpath_list = glob.glob(src_dir + '/*.' + ext)
        for i in range(len(fullpath_list)):
            fullpath_list[i] = fullpath_list[i].replace("\\","/")
        done_fullpath_list = glob.glob(out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))

    if level == 2:
        vid_list = list(
            map(lambda p: os.path.join('/'.join(p.split('/')[-2:])), fullpath_list))
    elif level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))

    pool = Pool(num_worker)
    pool.map(dump_frames, zip(fullpath_list, vid_list, range(len(vid_list))))


if __name__ == '__main__':
    extract_frames()  # 首次运行请取消注释
