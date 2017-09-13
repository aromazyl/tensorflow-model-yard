#! /bin/sh
#
# run.sh
# Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.
#


last_model=$1
if [ $last_model"1" == "1" ];then
  last_model=empty
fi

python mnet_train.py --train_dir ./tfrecords \
  --style_img_path ./style_images/surreal.png \
  --model_name surreal_v1 \
  --n_epochs 2 \
  --batch_size 4 \
  --content_weights 2.5 \
  --last_model $last_model \
  --style_weights 5.0 5.0 5.0 5.0 \
  --beta 1.e-4 \
  --style_target_resize 0.5
