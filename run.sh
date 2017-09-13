#! /bin/sh
#
# run.sh
# Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.
#


python train.py --train_dir ./tfrecords \
  --style_img_path ./style_images/bricks.jpg \
  --model_name bricks \
  --n_epochs 2 \
  --batch_size 4 \
  --content_weights 0.5 \
  --style_weights 5.0 5.0 5.0 5.0 \
  --beta 1.e-4 \
  --style_target_resize 0.5
