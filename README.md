# 3DDUNET
This is the code for 3D2Unet: 3D Deformable Unet for Low-Light Video Enhancement (PRCV2021) 
[Conference Paper Link](https://link.springer.com/book/10.1007/978-3-030-88004-0) 


## Dataset

We use SMOID dataset from [SMOID](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark)

## Code


### Prerequisites

- Python 3.6
- PyTorch 1.7 with GPU
- opencv-python
- scikit-image
- tensorboard

### Train and Test

Please run main.py to train and test the model

## Citing

If you use any part of our research, please consider citing:

```bibtex
@inproceedings{zeng2021mathrm,
  title={3D2Unet:3D Deformable Unet for Low-Light Video Enhancement},
  author={Zeng, Yuhang and Zou, Yunhao and Fu, Ying},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={66--77},
  year={2021},
  organization={Springer}
}

```


## Acknowledgement
Our work and implementations are inspired by following projects:
[ESTRNN](https://github.com/zzh-tech/ESTRNN)
[SMOID](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark)
