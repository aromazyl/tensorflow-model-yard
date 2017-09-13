#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 zhangyule <zyl2336709@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
from PIL import Image
import os
import sys
import scipy.misc
import cv2
import numpy as np
import glob
from multiprocessing import Pool
from resizeimage import resizeimage

np.set_printoptions(threshold=np.nan)

def ReadImage(image):
    img = cv2.imread(image)
    if image.endswith('jpg') or image.endswith('JPG') or image.endswith('bmp'):
        img[:,:,[0,2]] = img[:,:,[2,0]]
    return img

def ConvertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def ConvertFromBGRToHSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def ConvertFromHSVToBGR(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def GetImageGray(image):
    return ConvertToGray(ReadImage(image))

def GetImageRGBA(image):
    # return Image.open(image, 'r').split()
    return ReadImage(image)

def GetImageR(image):
    return GetImageRGBA(image)[:,:,0]

def GetImageG(image):
    return GetImageRGBA(image)[:,:,1]

def GetImageB(image):
    return GetImageRGBA(image)[:,:,2]

def GetImageA(image):
    return GetImageRGBA(image)[:,:,3]

def GetImageFlow(image_prev, image_curr):
    return cv2.calcOpticalFlowFarneback(image_prev, image_curr, 0.5, 3, 15, 3, 5, 1.2, 0)

def MapToElements(ndArray, func):
    func2 = np.vectorize(func)
    return func2(ndArray)

def GetCorrespondingImage(image_path, path1, path2):
    return path2 + image_path[len(path1):-3] + "png"

def GetSubPathImages(path):
    return glob.glob(path + "/*")


def GetCorrespondingImages(path1, path2):
    folders1 = glob.glob(path1 + "/*")
    ret = []
    for sub_path in folders1:
        sub_path_images = []
        m = glob.glob(sub_path + "/*")
        for elem in m:
            file_path = GetCorrespondingImage(elem, path1, path2)
            if os.path.isfile(file_path):
                sub_path_images.append((elem, file_path))
            else:
                print file_path, 'is not exist!'
        ret.append(sorted(sub_path_images))
    return ret

def PoolExecute(func, lists):
    pool = Pool(processes=8)
    pool.map(func, lists)

def SaveNpyImage(image, path):
    scipy.misc.imsave(path, image)

def DumpNumpy(arr):
    print arr

def WarpImage(src, flow):
    w, h, _ = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[:, y, 0] = float(y) + flow[:, y, 0]
    for x in range(w):
        flow_map[x, :, 1] = float(x) + flow[x, :, 1]
    dst = cv2.remap(
        src, flow_map[:,:,0], flow_map[:,:,1],
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def ResizeImage(src, shape, save_path = None):
    image = Image.open(src)
    cover = resizeimage.resize_cover(image, shape)
    if save_path != None:
        cover.save(save_path, image.format)
    return cover

def PixelConvertRgb2Hsv(r,g,b):
    r_,g_,b_ = r/255., g/255., b/255.
    cmax = max(r_,g_,b_)
    cmin = min(r_,g_,b_)
    delta = float(cmax - cmin)
    hue = 0.0
    if delta == 0:
        hue = 0
    elif cmax == r_:
        hue = int((g_ - b_) / delta) % 6
    elif cmax == g_:
        hue = ((b_ - r_) / delta) + 2
    else:
        hue = ((r_ - g_) / delta) + 4
    hue = hue * 60
    saturation = 0.0 if cmax == 0 else delta / cmax
    value = cmax
    return np.array(hue, saturation, value)

def PixelConvertHsv2Rgb(h, s, v):
    c = v * s
    x = c * (1.0 - abs(int(h / 60) % 2 - 1))
    m = v - c
    rgb_ = lambda:((c,x,0), (x,c,0), (0,c,x), (0,x,c), (x, 0, c), (c, 0, x))
    rgb_ = np.array(rgb_()[(int)(h / 60)])
    return (rgb_ + m) * 255

def PoolExcecute(func, lists):
    pool = Pool(processes=8)
    pool.map(func, lists)


if __name__ == '__main__':
    # print GetCorrespondingImages(sys.argv[1], sys.argv[2])
    # SaveNpyImage(GetImageGray(sys.argv[1]), sys.argv[2])
    # print GetImageFlow(GetImageGray(sys.argv[1]), GetImageGray(sys.argv[2])).shape
    for image in GetSubPathImages(sys.argv[1]):
        rate = 0.3
        img = ReadImage(image)
        oimg = ReadImage(sys.argv[2] + "/" + image.split("/")[-1])
        hsv = ConvertFromBGRToHSV(img)
        ohsv = ConvertFromBGRToHSV(oimg)
        hsv[:,:,2] = hsv[:,:,2] * (1.0-rate) + ohsv[:,:,2] * rate
        # img = 0.5 * img + 0.5 * oimg
        SaveNpyImage(ConvertFromHSVToBGR(hsv), "hsv_result/" + image.split("/")[-1])
    # ResizeImage(sys.argv[1], (256,256), sys.argv[2])
