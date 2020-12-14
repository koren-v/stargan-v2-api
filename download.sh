"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

URL=https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt?dl=0
mkdir -p ./checkpoints/
OUT_FILE=./checkpoints/wing.ckpt
wget -N $URL -O $OUT_FILE
URL=https://www.dropbox.com/s/91fth49gyb7xksk/celeba_lm_mean.npz?dl=0
OUT_FILE=./checkpoints/celeba_lm_mean.npz
wget -N $URL -O $OUT_FILE

URL=https://www.dropbox.com/s/96fmei6c93o8b8t/100000_nets_ema.ckpt?dl=0
mkdir -p ./checkpoints/celeba_hq
OUT_FILE=./checkpoints/celeba_hq/100000_nets_ema.ckpt
wget -N $URL -O $OUT_FILE

URL=https://www.dropbox.com/s/etwm810v25h42sn/100000_nets_ema.ckpt?dl=0
mkdir -p ./checkpoints/afhq
OUT_FILE=./checkpoints/afhq/100000_nets_ema.ckpt
wget -N $URL -O $OUT_FILE
