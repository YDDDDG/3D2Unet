import cv2
import numpy as np
import os

split = ['Dark20190314-Place1', 
        'Dark20190314-Place2',
        'Dark20190314-Place5',
        'Dark20181113-Skylight']
rootdir = '/data/zengyuhang_data/0_data'

def unpack(vid):
    new_vid = np.zeros([vid.shape[0], vid.shape[1] * 2, vid.shape[2] * 2], dtype='uint16')
    new_vid[:, ::2, ::2] = vid[:, :, :, 0]
    new_vid[:, ::2, 1::2] = vid[:, :, :, 1]
    new_vid[:, 1::2, ::2] = vid[:, :, :, 2]
    new_vid[:, 1::2, 1::2] = vid[:, :, :, 3]
    return new_vid

def demosaic(dark, converter=cv2.COLOR_BAYER_GR2BGR):
   new_vid = np.zeros([dark.shape[0], dark.shape[1], dark.shape[2], 3])
   for i in range(dark.shape[0]):
       new_vid[i] = cv2.cvtColor(dark[i], converter)
   return new_vid

for s in split:
    print('for part:', s)
    #train = os.path.join(rootdir, s, 'train')
    gt = os.path.join(rootdir, s, 'gt')
    #print('train dir:', train)
    print('gt dir', gt)
    #train_fs = os.listdir(train)
    gt_fs = os.listdir(gt)
    gt_fs.sort()

    #assert len(train_fs) == len(gt_fs), 'len mismatch'
    #print('train file number:', len(train_fs))
    print('gt file number:', len(gt_fs))

    #train_path = [os.path.join(train, f) for f in train_fs]
    gt_path = [os.path.join(gt, f) for f in gt_fs]

    for index in range(len(gt_path)):
        #train_fn = train_path[index]
        gt_fn = gt_path[index]

        print('load files')
        #train_array = np.load(train_fn, mmap_mode='r')
        #print('train loaded')
        gt_array = np.load(gt_fn)
        print('gt loaded')

        #train_array = demosaic(unpack(train_array))
        gt_array = np.flip(gt_array, axis=3)
        #print(train_array.shape, gt_array.shape)     

        for i in range(200):
            #img_train = train_array[i, :, :, :]
            img_gt = gt_array[i, :, :, :]
            #img_train = np.rot90(img_train, 2, axes=(0, 1))
            img_gt = np.rot90(img_gt, 2, axes=(0, 1))

            #cv2.imwrite('/data/zhangfan/data_dark_alignment/imgdata/train/'+s+'_video'+str(index)+'_frame'+str(i)+'.png', img_train)
            cv2.imwrite('/data/zengyuhang_data/0_data/image_data/gt/'+s+'_video'+str(index)+'_frame'+str(i)+'.png', img_gt)
            print('frame '+str(i)+ 'completed')

        print('video '+str(index)+' completed')

    print('split '+s+' completed')



