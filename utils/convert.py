import cv2
import numpy as np
import os

split = ['Dark20190314-Place1', 
        'Dark20190314-Place2',
        'Dark20190314-Place5',
        'Dark20181113-Skylight']
rootdir =  '/dataset_npy/0_data'

def unpack(vid):
    new_vid = np.zeros([vid.shape[0], vid.shape[1] * 2, vid.shape[2] * 2], dtype='uint16')
    new_vid[:, ::2, ::2] = vid[:, :, :, 0]    #g
    new_vid[:, ::2, 1::2] = vid[:, :, :, 1]   #r
    new_vid[:, 1::2, ::2] = vid[:, :, :, 2]    #b
    new_vid[:, 1::2, 1::2] = vid[:, :, :, 3]    #g
    return new_vid


def restored_to_raw(img):
    raw_img = np.zeros([1, img.shape[1] * 2, img.shape[2] * 2], dtype=np.uint16)
    r=img[0]
    g1=img[1]
    b=img[2]
    g2=img[3]
    for i in range( img.shape[1]):
        for j in range(img.shape[2]):
            raw_img[0][i*2][j*2]=r[i][j]
            raw_img[0][i*2][j*2+1]=g1[i][j]
            raw_img[0][i*2+1][j*2]=g2[i][j]
            raw_img[0][i*2+1][j*2+1]=b[i][j]
    raw_img = raw_img.transpose(1, 2, 0)
    return raw_img

def demosaic(dark, converter=cv2.COLOR_BAYER_GR2BGR):
   new_vid = np.zeros([dark.shape[0], dark.shape[1], dark.shape[2], 3])
   for i in range(dark.shape[0]):
       new_vid[i] = cv2.cvtColor(dark[i], converter)
   return new_vid


def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)

if __name__ == '__main__':
    for s in split:
        print('for part:', s)
        train = os.path.join(rootdir, s, 'train')
        #gt = os.path.join(rootdir, s, 'gt')
        print('train dir:', train)
        #print('gt dir', gt)
        train_fs = os.listdir(train)
        train_fs.sort()
        #gt_fs = os.listdir(gt)

        #assert len(train_fs) == len(gt_fs), 'len mismatch'
        print('train file number:', len(train_fs))
        #print('gt file number:', len(gt_fs))

        train_path = [os.path.join(train, f) for f in train_fs]
        #gt_path = [os.path.join(gt, f) for f in gt_fs]

        for index in range(len(train_path)):
            train_fn = train_path[index]
            #gt_fn = gt_path[index]

            print('load files')
            train_array = np.load(train_fn)*15
            print('train loaded')
            #gt_array = np.load(gt_fn, mmap_mode='r')
            #print('gt loaded')
            print(train_array.shape)
            train_array = unpack(train_array).clip(0,65535)
            print("min and max are~~~~~~~~~~~~~~~~",train_array.min(),train_array.max())


            # print("equalize_histogram start")
            # print(train_array.mean())
            # train_array=equalize_histogram(train_array,65536).clip(0,65535)
            # print(train_array.mean())


            # train_array = demosaic(train_array)
            #gt_array = np.flip(gt_array, axis=3)
            #print(train_array.shape, gt_array.shape)     

            for i in range(200):
                print("train_array.shape",train_array.shape)
                img_train = train_array[i, :, :]
                #img_gt = gt_array[i, :, :, :]
                print('img_train.shape',img_train.shape)
                
                img_train = np.rot90(img_train, 2, axes=(0, 1))

                cv2.imwrite('/data/SMOID/0_data/image_data/train/'+s+'_video'+str(index)+'_frame'+str(i)+'.tiff', img_train.astype(np.uint16))
                print('frame '+str(i)+ 'completed')

            print('video '+str(index)+' completed')

        print('split '+s+' completed')



