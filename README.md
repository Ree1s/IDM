<TOC>

# Implicit Diffusion Models for Continuous Super-Resolution

This repository is an offical implementation of the paper "Implicit Diffusion Models for Continuous Super-Resolution" from CVPR 2023.

This repository is still under development.


## Environment configuration

The codes are based on python3.7+, CUDA version 11.0+. The specific configuration steps are as follows:

1. Create conda environment
   
   ```shell
   conda create -n idm python=3.7.10
   conda activate idm
   ```

2. Install pytorch
   
   ```shell
   conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3
   ```

3. Installation profile
   
   ```shell
   pip install -r requirements.txt
   python setup.py develop
   ```
## Data preparation
Firstly, download the datasets used.
- [FFHQ](https://github.com/NVlabs/ffhq-dataset) | [CelebaHQ](https://www.kaggle.com/badasstechie/celebahq-resized-256x256)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) ｜ [Flick2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

Then, resize to get LR_IMGS and HR_IMGS.
```
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```
## Pre-trained checkpoints

The pre-trained checkpoints can be found at the following: [link](https://drive.google.com/drive/folders/1VISy9fVWa9iOSr6F4oVtKVTOViWuKohQ?usp=drive_link).

## Training and Validation
Run the following command for the training and validation:

   ```shell
   sh run.sh
   ```
Add the command "-use_ddim" to implement DDIM sampling.

## Acknowledgements
This code is mainly built on [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), and [LIIF](https://github.com/yinboc/liif).