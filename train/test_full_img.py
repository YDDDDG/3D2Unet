

##  python main.py --test_only --gain 00 --test_checkpoint /data/zengyuhang_data/Result_Unet-raw2rgb/experiment/2021_04_04_20_50_32_Unet_LowLight/checkpoint.pth.tar




import os
import pickle
import time
from os.path import join, dirname

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn

from data.utils import normalize, normalize_reverse,get_file_path
from model import Model
from .metrics import psnr_calculate, ssim_calculate, psnr_calculate2
from .utils import AverageMeter, img2video
from utils.show_imgs import ISP_4channels_and_not_show, restored_to_raw, equalize_histogram


#dark_img.shape
#(1, 4, 518, 939)
#gt_img.shape
#(1036, 1878, 3)

def crop_centre(raw_img, rgb_img, crop_size=(512,512)):
    b, c, x, y = raw_img.shape
    startx = x // 2 - (crop_size[0] // 2 )
    straty = y // 2 - (crop_size[1] // 2 )

    raw=raw_img[ :, :, startx:startx+crop_size[0],straty:straty+crop_size[1] ] 
    rgb=rgb_img[startx*2 : (startx+crop_size[0])*2 , straty*2 : (straty+crop_size[1])*2 , :]

    return raw, rgb

def unpack(vid):
    new_vid = np.zeros([1, vid.shape[1] * 2, vid.shape[2] * 2], dtype='uint16')
    new_vid[:, ::2, ::2] = vid[0, :, :]
    new_vid[:, ::2, 1::2] = vid[1, :, :]
    new_vid[:, 1::2, ::2] = vid[2, :, :]  #2
    new_vid[:, 1::2, 1::2] = vid[3, :, :]  #3
    return new_vid


def test(para, logger):
    """
    test code
    """
    # load model with checkpoint
    if not para.test_only:
        para.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if para.test_save_dir is None:
        para.test_save_dir = logger.save_dir
    model = Model(para).cuda()
    checkpoint_path = para.test_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])

    ds_name = para.dataset
    logger('{} results generating ...'.format(ds_name), prefix='\n')
    if ds_name == 'BSD':
        ds_type = 'test'
        _test_torch(para, logger, model, ds_type)
    elif ds_name=='LowLight':
        ds_type = 'test'
        _test_torch(para, logger, model, ds_type)
    elif ds_name == 'gopro_ds_lmdb' or ds_name == 'reds_lmdb':
        ds_type = 'valid'
        _test_lmdb(para, logger, model, ds_type)
    else:
        raise NotImplementedError


def _test_torch(para, logger, model, ds_type):
    PSNR_1 = AverageMeter()
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()



    dataset_path = join(para.data_root)


    _ids, gt_files, in_files = get_file_path(ds_type, para.gain ,dataset_path)


    seq_length=200
    for seq in _ids:

        print("sequence:",seq,"start!")

        video=seq[-4:]
        gain=seq[-7:-5]
        place=seq[:-12]
        path=dataset_path+place+'/'+gain+'/'+video+'/'


        logger('seq {} image results generating ...'.format(seq))
        torch.cuda.empty_cache()
        dir_name = '_'.join((para.dataset, para.model, 'test'))
        save_dir = join(para.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = para.test_frames                                            ###########测试帧数，注意开始与结束的设置 ，frame
        while True:
            input_seq = []
            label_seq = []
            for frame_idx in range(start, end):
                dark_img_path = join(path, 'train', '{:d}.{}'.format(frame_idx, 'tiff'))
                bright_img_path = join(path, 'gt', '{:d}.{}'.format(frame_idx, 'png'))

                dark_img = cv2.imread(dark_img_path,-1)[..., np.newaxis].astype(np.int32)

                # dark_img=equalize_histogram(dark_img, number_bins=65536)

                img_shape = dark_img.shape
                H = img_shape[0]
                W = img_shape[1]
                im = np.clip(dark_img,0,65535)
                im = np.concatenate((im[0:H:2, 0:W:2, :],
                                    im[0:H:2, 1:W:2, :],
                                    im[1:H:2, 0:W:2, :],
                                    im[1:H:2, 1:W:2, :]), axis=2)
                dark_img = im.transpose(2, 0, 1)[np.newaxis, ...]

                gt_img=cv2.imread(bright_img_path)
  
                dark_img, gt_img= crop_centre(dark_img, gt_img)

                input_seq.append(dark_img)
                label_seq.append(gt_img)
                


            #(1, 4, 512, 512)

            input_seq = np.concatenate(input_seq)[np.newaxis, :]
            #(1, 16, 4, 512, 512)
            model.eval()
            with torch.no_grad():
                input_seq = normalize(torch.from_numpy(input_seq).float().cuda(),centralize=False,
                                    normalize=True, val_range=65535).permute(0,2,1,3,4)
                time_start = time.time()
                output_seq = model([input_seq]).squeeze(dim=0)
                timer.update(time.time() - time_start, n=len(output_seq))
                input_seq=input_seq.permute(0,2,1,3,4)
                output_seq=output_seq.permute(1,0,2,3)
                


            #第一个循环：range(2,21-1-2)->(2,18)->2,3,4,5,....17->16次循环
            for frame_idx in range(0, para.test_frames):
                dark_img = input_seq.squeeze(dim=0)[frame_idx]
                dark_img = normalize_reverse(dark_img, centralize=False, normalize=True,val_range=65535.0)
                #dark_img = dark_img.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                dark_img = dark_img.detach().cpu().numpy().astype(np.uint16)
                dark_img=unpack(dark_img)
                dark_img=dark_img.transpose(1,2,0)
                dark_img_path = join(save_dir, '{:d}_input.tiff'.format(frame_idx + start))

                gt_img = label_seq[frame_idx]
                # gt_img=gt_img.transpose(2, 0, 1)
                gt_img_path = join(save_dir, '{:d}_gt.png'.format(frame_idx + start))
                restored_img = output_seq[frame_idx]
                restored_img = normalize_reverse(restored_img, centralize=False, normalize=True,val_range=255.0)
                #restored_img = restored_img.detach().cpu().numpy().transpose((1, 2, 0))
                restored_img = restored_img.detach().cpu().numpy()
                restored_img=np.clip(restored_img,0,255).round().astype(np.uint8).transpose(1,2,0)
                restored_img_path = join(save_dir, '{:d}_{}.png'.format(frame_idx + start, para.model.lower()))

                # calc_psnr_as_rgb = False
                # if calc_psnr_as_rgb:
                #     #将4通道raw可视化为rgb
                #     dark_img,_=ISP_4channels_and_not_show(dark_img,gt_img)
                #     restored_img,gt_img=ISP_4channels_and_not_show(restored_img,gt_img)
                # else:
                #     #将4通道转化为1通道raw
                #     dark_img=restored_to_raw(dark_img)
                #     restored_img=restored_to_raw(restored_img)
                #     gt_img=restored_to_raw(gt_img)

                #     dark_img_path = join(save_dir, '{:d}_input.tiff'.format(frame_idx + start))
                #     gt_img_path = join(save_dir, '{:d}_gt.tiff'.format(frame_idx + start))
                #     restored_img_path = join(save_dir, '{:d}_{}.tiff'.format(frame_idx + start, para.model.lower()))



                cv2.imwrite(dark_img_path, dark_img)
                cv2.imwrite(gt_img_path, gt_img)
                cv2.imwrite(restored_img_path, restored_img)
                if restored_img_path not in results_register:
                    results_register.add(restored_img_path)
                    psnr=psnr_calculate(restored_img, gt_img)
                    ssim=ssim_calculate(restored_img, gt_img)


                    psnr_1=psnr_calculate2(restored_img, gt_img,255.0)

                    PSNR_1.update(psnr_1)
                    PSNR.update(psnr)
                    SSIM.update(ssim)
                    print("restored_img_path's psnr,psnr_1 and ssim are:",psnr,psnr_1,ssim)
                    # if calc_psnr_as_rgb:
                    #     PSNR.update(psnr_calculate(restored_img, gt_img))
                    #     SSIM.update(ssim_calculate(restored_img, gt_img))
                    # else:
                    #     PSNR.update(psnr_calculate(restored_img, gt_img,65535.0))
                    #     SSIM.update(ssim_calculate(restored_img, gt_img,65535))
            
            print("start is",start)

            if end >= seq_length:
                break
            else:
                start = end
                end = end + para.test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - para.test_frames
        
        # print("start making a video")

        # if para.video:
        #     logger('seq {} video result generating ...'.format(seq))
        #     marks = ['Input', para.model, 'GT']
        #     path = dirname(save_dir)

        #     W=1024
        #     H=1024
        #     img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=3, frame_end=199+1,
        #               marks=marks, fps=10)
        
        # # print("end making a video")

        # print(seq,"end!")
    


    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test PSNR_1 : {}'.format(PSNR_1.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))
