#! /bin/sh
#
# run.sh
# Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.
#


python flow_train.py --train_dir ~/fast-neural-style-flow-tensorflow/fast-neural-style-tensorflow/DAVIS/512x512p/ \
  --style_img_path ./style_images/bricks.jpg \
  --model_name flow_low_filter_bricks_v2 \
  --n_epochs 2 \
  --batch_size 2 \
  --content_weights 3 \
  --temporal_weight 100 \
  --style_weights 5.0 5.0 5.0 5.0 \
  --beta 1.e-4 \
  --style_target_resize 0.5 \
  --last_model $1 \
