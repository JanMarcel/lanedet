#!usr/bin/env bash
echo $1
python tools/detect.py configs/laneatt/mobilenetv2_tusimple_100epochs_maxlanes2.py --load_from work_dirs/TuSimple/20240519_170433_lr_3e-04_b_8/ckpt/$1.pth --img data/Labor/ --savedir vis/mobile_tusimple_$1_new_img