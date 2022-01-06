import cv2
import numpy as np
import os

# split = ['Dark20190314-Place1', 
#         'Dark20190314-Place2',
#         'Dark20190314-Place5',
#         'Dark20181113-Skylight']

split = [
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

def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)


# generate raw images
if __name__ == '__main__':
    for s in split:
        print('for part:', s)
        train = os.path.join(rootdir, s, 'train')
        print('train dir:', train)
        train_fs = os.listdir(train)
        train_fs.sort()

        print('train file number:', len(train_fs))


        train_path = [os.path.join(train, f) for f in train_fs]


        for index in range(len(train_path)):
            train_fn = train_path[index]

            gain=train_fn[-11:-9]

            video=train_fn[-8:-4]

            print('load files')
            train_array = np.load(train_fn)
            print('train loaded')


            train_array = unpack(train_array)


            # print("equalize_histogram start")
            # print(train_array.mean())
            # train_array=equalize_histogram(train_array,65536).clip(0,65535)
            # print(train_array.mean())

            # print("equalize_histogram end")
  

            for i in range(200):
                img_train = train_array[i, :, :]

                img_train = np.rot90(img_train, 2, axes=(0, 1))
                dst='/data/zengyuhang_data/0_data/processed_imgs/'
                detaild_dst =s+'/'+gain + '/'+video+'/train/'
                os.makedirs(dst+detaild_dst,exist_ok=True)
                cv2.imwrite(dst+detaild_dst+str(i)+'.tiff', img_train.astype(np.uint16))

                print('frame '+str(i)+ 'completed')

            print('video '+str(index)+' completed')

        print('split '+s+' completed')


# #generate rgb images
# if __name__ == '__main__':
#     for s in split:
#         print('for part:', s)
#         gt = os.path.join(rootdir, s, 'gt')
#         print('gt dir:', gt)
#         gt_fs = os.listdir(gt)
#         gt_fs.sort()


#         #assert len(train_fs) == len(gt_fs), 'len mismatch'

#         print('gt file number:', len(gt))


#         gt_path = [os.path.join(gt, f) for f in gt_fs]


#         for index in range(len(gt_path)):
#             gt_fn = gt_path[index]

#             gain=gt_fn[-11:-9]

#             video=gt_fn[-8:-4]

#             print('load files')
#             gt_array = np.load(gt_fn)
#             print('train loaded')

#             gt_array = np.flip(gt_array, axis=3)

#             for i in range(200):
#                 img_gt = gt_array[i, :, :, :]

#                 img_gt = np.rot90(img_gt, 2, axes=(0, 1))
#                 dst='/data/zengyuhang_data/0_data/processed_imgs/'
#                 detaild_dst =s+'/'+gain + '/'+video+'/gt/'
#                 os.makedirs(dst+detaild_dst,exist_ok=True)
#                 cv2.imwrite(dst+detaild_dst+str(i)+'.png', img_gt.astype(np.uint8))

#                 print('frame '+str(i)+ 'completed')

#             print('video '+str(index)+' completed')

#         print('split '+s+' completed')
