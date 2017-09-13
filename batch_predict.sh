#! /bin/sh
#
# predict.sh
# Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.
#


python batch_predict.py --input_img_path $1 \
                        --output_img_path ./result/ \
                        --model_path ./$2

