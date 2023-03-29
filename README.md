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
## Pre-trained checkpoint

The pre-trained checkpoint and the val dataset of face 8X SR can be found at the following anonymous link: [link](https://1drv.ms/u/s!AraiW_uJqO8vhnlIa-8nd0PEH4Ur?e=qDfSep). Download and unzip `checkpoint_dataset.zip`. Move `checkpoint_dataset/best_psnr_gen.pth` and `checkpoint_dataset/dataset` to `./`.

## Validation
Run the following command for the validation:

   ```shell
   sh run.sh
   ```

## Acknowledgements
This code is mainly built on [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), and [LIIF](https://github.com/yinboc/liif).