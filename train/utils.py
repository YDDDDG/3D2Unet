import os
from os.path import join, dirname

import cv2
import numpy as np
import torch.distributed as dist


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# todo in case of dict
def reduce_tensor(num_gpus, ts):
    """
    reduce tensor from multiple gpus
    """
    # todo loss of ddp mode
    if isinstance(ts, dict):
        raise NotImplementedError
    else:
        try:
            dist.reduce(ts, dst=0, op=dist.ReduceOp.SUM)
            ts /= num_gpus
        except:
            msg = '{}'.format(type(ts))
            raise NotImplementedError(msg)
    return ts


def img2video(path, size, seq, frame_start, frame_end, marks, fps=10):
    """
    generate video
    """
    file_path = join(path, '{}.avi'.format(seq))
    os.makedirs(dirname(path), exist_ok=True)
    path = join(path, '{}'.format("video"+seq))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")


    #get iamge size
    sample_img_path=join(path, '{:d}_{}.png'.format(1, 'gt'))
    img = cv2.imread(sample_img_path)
    size=(img.shape[1]*3, img.shape[0])


    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for i in range(frame_start, frame_end):
        imgs = []
        for j in range(len(marks)):
            img_path = join(path, '{:d}_{}.png'.format(i, marks[j].lower()))
            print("~~~~~~~~~~~~~~~~~~img path is:",img_path)
            img = cv2.imread(img_path)
            print("img_path,shape is:",img.shape)
            img = cv2.putText(img, marks[j], (60, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            imgs.append(img)
        print("oh~~~~there is the imgs' shape:",(np.array(imgs)).shape)
        frame = np.concatenate(imgs, axis=1)
        video.write(frame)
    video.release()


def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)