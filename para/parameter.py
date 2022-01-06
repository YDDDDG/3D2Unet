import argparse


#train  python main.py

#resume nohup python main.py --resume  --resume_file './experiment/2021_01_05_21_00_02_ESTRNN_LowLight/checkpoint.pth.tar'> out.log 2>&1 &

#test  python main.py --test_only   --test_frames 20   --test_checkpoint ./model_best.pth.tar  --video

#test the speed or parameters  python main.py  --profile_H 480  --profile_W 640


class Parameter:
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser(description='Video Low Light Enhancement')

        #设置实验的Gain值
        self.parser.add_argument('--gain', type=str, default='00', help=" gain = '00' or '05' or '10' or '15' or '20' ")

        # experiment mark
        self.parser.add_argument('--description', type=str, default='develop', help='experiment description')

        # global parameters
        self.parser.add_argument('--seed', type=int, default=39, help='random seed')
        self.parser.add_argument('--threads', type=int, default=4, help='# of threads for dataloader')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
        self.parser.add_argument('--no_profile', action='store_false', help='show # of parameters and computation cost')
        self.parser.add_argument('--profile_H', type=int, default=512,
                                 help='height of image to generate profile of model')
        self.parser.add_argument('--profile_W', type=int, default=512,
                                 help='width of image to generate profile of model')
        self.parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_file', type=str, default='/data/zengyuhang_data/Result_Unet-raw2rgb/experiment/2021_04_16_01_06_06_Unet_LowLight/checkpoint.pth.tar', help='the path of checkpolossint file for resume')

        # data parameters
        #self.parser.add_argument('--data_root', type=str, default='H:\\Unzipped_Processed_Data\\SMOID_ND01_15fps\\', help='the path of dataset')
        self.parser.add_argument('--data_root', type=str, default='/data/zengyuhang_data/processed_imgs/',help='the path of dataset')
        self.parser.add_argument('--dataset', type=str, default='LowLight', help='name is Lowlight')


        self.parser.add_argument('--save_dir', type=str, default='/data/zengyuhang_data/Result_Unet-raw2rgb/experiment/',
                                 help='directory to save logs of experiments')
        self.parser.add_argument('--frames', type=int, default=16, help='# of frames of subsequence')
        self.parser.add_argument('--data_format', type=str, default='RAW', help='RGB or RAW')
        self.parser.add_argument('--patch_size', type=list, nargs='*', default=[256, 256])

        # model parameters
        self.parser.add_argument('--model', type=str, default='Unet', help='type of model to construct')
        self.parser.add_argument('--n_features', type=int, default=16, help='base # of channels for Conv')
        self.parser.add_argument('--n_blocks', type=int, default=12, help='# of blocks in middle part of the model')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation function')

        # loss parameters
        self.parser.add_argument('--loss', type=str, default='1*L1',
                                 help='type of loss function, e.g. 1*MSE|1e-4*Perceptual')

        # metrics parameters
        self.parser.add_argument('--metrics', type=str, default='PSNR', help='type of evaluation metrics')

        # optimizer parameters
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='method of optimization')
        self.parser.add_argument('--lr', type=float, default=5e-5, help='learning rate') 
        self.parser.add_argument('--lr_scheduler', type=str, default='multi_step',
                                 help='learning rate adjustment stratedy :consine, multi_step')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--milestones', type=int, nargs='*', default=[])
        self.parser.add_argument('--decay_gamma', type=float, default=0.1, help='decay rate')

        # training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=500, help='last epoch number')
        self.parser.add_argument('--trainer_mode', type=str, default='dp',
                                 help='trainer mode: distributed data parallel (ddp) or data parallel (dp)')

        # test parameters
        self.parser.add_argument('--test_only', action='store_true', help='only do test')
        self.parser.add_argument('--test_frames', type=int, default=32,
                                 help='frame size for test, if GPU memory is small, please reduce this value')
        self.parser.add_argument('--test_save_dir',default='/data/zengyuhang_data/Result_Unet-raw2rgb/result/', type=str, help='where to save test results')
        self.parser.add_argument('--test_checkpoint', type=str,default='/data/zengyuhang_data/Result_Unet-raw2rgb/experiment/2021_04_13_16_49_46_Unet_LowLight/model_best.pth.tar',
                                 help='the path of checkpoint file for test')
        self.parser.add_argument('--video', action='store_false', help='if true, generate video results')

        args, _ = self.parser.parse_known_args()

        return args
