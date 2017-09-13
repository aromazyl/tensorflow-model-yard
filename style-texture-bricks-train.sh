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

python texture_net_train.py --train_dir ./tfrecords \
  --style_img_path ./style_images/bricks.jpg \
  --model_name texture_net_bricks \
  --n_epochs 2 \
  --batch_size 4 \
  --content_weights 3 \
  --last_model $last_model \
  --style_weights 5.0 5.0 5.0 5.0 \
  --beta 1.e-4 \
  --style_target_resize 0.5
