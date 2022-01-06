import numpy as np
import torch
import gc

class Crop(object):
    """
    Crop randomly the image in a sample.
    Args: output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        top, left = sample['top'], sample['left']
        new_h, new_w = self.output_size
        new_h//=2
        new_w//=2
        sample['image'] = image[top: top + new_h,
                          left: left + new_w,:]
        sample['label'] = label[top*2: (top + new_h)*2,
                          left*2: (left + new_w)*2,:]

        return sample


class Flip(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        flag_ud = sample['flip_ud']
        if flag_lr == 1:
            sample['image'] = np.fliplr(sample['image'])
            sample['label'] = np.fliplr(sample['label'])
        if flag_ud == 1:
            sample['image'] = np.flipud(sample['image'])
            sample['label'] = np.flipud(sample['label'])

        return sample


class Rotate(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class Sharp2Sharp(object):
    def __call__(self, sample):
        flag = sample['s2s']
        if flag < 1:
            sample['image'] = sample['label'].copy()
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
        label = np.ascontiguousarray(label.transpose((2, 0, 1))[np.newaxis, :])
        sample['image'] = torch.from_numpy(image).float()
        sample['label'] = torch.from_numpy(label).float()
        return sample


def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x

def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)

def get_file_path(ds_type, gain='00', root_path='/data/zengyuhang_data'):

        ds_type=ds_type
        FILE_LIST="./data_list/"+ ds_type + "_list"
                # get train IDs
        with open(FILE_LIST) as f:
                text = f.readlines()
        _files = text

        _ids = [line.strip().split(' ')[0] for line in _files]
        gt_files = [line.strip().split(' ')[1] for line in _files]
        in_files = [line.strip().split(' ')[2] for line in _files]


        gain=gain
        _ids_copy=[]
        for item in _ids:
                if item[-7:-5]==gain:
                        _ids_copy.append(item)
        _ids=_ids_copy


        root_path=root_path

        gt_files_copy=[]
        for item in gt_files:
                if item[-11:-9]==gain:
                        gt_files_copy.append(root_path+item[1:])
        gt_files=gt_files_copy

        in_files_copy=[]
        for item in in_files:
                if item[-11:-9]==gain:
                        in_files_copy.append(root_path+item[1:])
        in_files=in_files_copy

        return _ids, gt_files, in_files


def get_all_file_path(ds_type, root_path='/data/zengyuhang_data'):

        ds_type=ds_type
        FILE_LIST="./data_list/"+ ds_type + "_list"
                # get train IDs
        with open(FILE_LIST) as f:
                text = f.readlines()
        _files = text

        _ids = [line.strip().split(' ')[0] for line in _files]
        gt_files = [line.strip().split(' ')[1] for line in _files]
        in_files = [line.strip().split(' ')[2] for line in _files]

        return _ids, gt_files, in_files


def gen_var(seqs):

    records=dict()

    for seq in seqs:
            sample=dict()


            print(seq[0],"start loading...")
            temp_dark=np.load(seq[0])
            dark_shape=temp_dark.shape
            del temp_dark
            gc.collect()

            print(seq[0],"start loading and EH process...")
            sample['Dark']=equalize_histogram(np.memmap(seq[0], dtype='uint16', mode='r',shape=dark_shape),65536)
            # sample['Dark']=equalize_histogram(np.load(seq[0]),65536)
            print(seq[0],"is processed")

            print(seq[1],"start loading...")
            temp_bright=np.load(seq[1])
            bright_shape=temp_bright.shape
            del temp_bright
            gc.collect()

            sample['Bright']=np.memmap(seq[1], dtype='uint8', mode='r',shape=bright_shape)
            # sample['Bright']=np.load(seq[1])
            print(seq[1],"is loaded")


            # print('dark_shape',sample['Dark'].shape)
            # print('bright_shape',sample['Bright'].shape)
            records[seq]=sample
    return records



def gen_seq(seq):

    sample=dict()


    print(seq[0],"start loading...")
    temp_dark=np.load(seq[0])
    dark_shape=temp_dark.shape
    del temp_dark
    gc.collect()

    print(seq[0],"start loading and EH process...")
    sample['Dark']=equalize_histogram(np.memmap(seq[0], dtype='uint16', mode='r',shape=dark_shape),65536)
    print(seq[0],"is processed")

    print(seq[1],"start loading...")
    temp_bright=np.load(seq[1])
    bright_shape=temp_bright.shape
    del temp_bright
    gc.collect()

    sample['Bright']=np.memmap(seq[1], dtype='uint8', mode='r',shape=bright_shape)
    print(seq[1],"is loaded")


    # print('dark_shape',sample['Dark'].shape)
    # print('bright_shape',sample['Bright'].shape)
    return sample
