import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import normalize, Crop, Flip, ToTensor ,get_all_file_path, get_file_path, equalize_histogram


class LowLightDataset(Dataset):

    def __init__(self, path, frames, crop_size=(256, 256), data_format='RAW', ds_type='train',gain ='00'):
        self.frames = frames
        self.data_format = data_format
        self.crop_h, self.crop_w = crop_size
        self.transform = transforms.Compose([Crop(crop_size), Flip(), ToTensor()])
        self._seq_length = 200
        self._samples = self._generate_samples(path, data_format, ds_type, gain)



    def _generate_samples(self, dataset_path, data_format, ds_type, gain):

        samples = list()
        records = dict()


        _ids, gt_files, in_files = get_file_path(ds_type, gain, dataset_path)

        for item in _ids:
            video=item[-4:]
            gain=item[-7:-5]
            place=item[:-12]

            
            path=dataset_path+place+'/'+gain+'/'+video+'/'

            records[item]=list()
            for frame in range(0,self._seq_length):
                sample = dict()
                sample['Dark'] = join(path, 'train','{:d}.{}'.format(frame, 'tiff'))
                sample['Bright'] = join(path, 'gt','{:d}.{}'.format(frame, 'png'))
                records[item].append(sample)

        for seq_records in records.values():
            temp_length = len(seq_records) - (self.frames - 1)
            if temp_length <= 0:
                raise IndexError('Exceed the maximum length of the video sequence')
            for idx in range(temp_length):
                samples.append(seq_records[idx:idx + self.frames])
        return samples






    def __getitem__(self, item):

        dark_imgs, bright_imgs = [], []
        for sample_dict in self._samples[item]:
            dark_img, bright_img = self._load_sample(sample_dict)
            dark_imgs.append(dark_img)
            bright_imgs.append(bright_img)
        dark_imgs = torch.cat(dark_imgs, dim=0).permute(1,0,2,3)
        bright_imgs =  torch.cat(bright_imgs, dim=0).permute(1,0,2,3)
        return dark_imgs, bright_imgs

    def _load_sample(self, sample_dict):

        raw_H=900
        raw_W=500

        if self.data_format == 'RAW':
            sample=dict()
            sample['image'] = cv2.imread(sample_dict['Dark'], -1)[..., np.newaxis].astype(np.int32)



            img_shape = sample['image'].shape
            H = img_shape[0]
            W = img_shape[1]
            im = sample['image']
            im = np.concatenate((im[0:H:2, 0:W:2, :],        #GBRG
                                 im[0:H:2, 1:W:2, :],
                                 im[1:H:2, 0:W:2, :],
                                 im[1:H:2, 1:W:2, :]), axis=2)
            raw_H=im.shape[0]
            raw_W=im.shape[1]
            sample['image'] = np.clip(im, 0, 65535)

        else:
            im = cv2.imread(sample_dict['Dark'])


        top = random.randint(0, raw_H - self.crop_h//2)
        left = random.randint(0, raw_W - self.crop_w//2)
        flip_lr = random.randint(0, 1)
        flip_ud = random.randint(0, 1)
        sample['top']= top
        sample['left']= left
        sample['flip_lr']= flip_lr
        sample['flip_ud']= flip_ud


        sample['label'] = cv2.imread(sample_dict['Bright'])


        sample = self.transform(sample)

        dark_img = normalize(sample['image'], centralize=False, normalize=True, val_range=65535.0)
        bright_img = normalize(sample['label'], centralize=False, normalize=True, val_range=255.0)

        return dark_img, bright_img

    def __len__(self):
        return len(self._samples)


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        path = join(para.data_root)
        frames = para.frames
        dataset = LowLightDataset(path, frames, para.patch_size, para.data_format, ds_type, para.gain)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler,
                drop_last=True
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True,
                drop_last=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len

if __name__ == '__main__':
    from para import Parameter

    para = Parameter().args
    para.data_format = 'RAW'
    para.dataset = 'LowLight'
    dataloader = Dataloader(para, 0)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
    print(x.type(), y.type())
    print(np.max(x.numpy()), np.min(x.numpy()))
    print(np.max(y.numpy()), np.min(y.numpy()))
