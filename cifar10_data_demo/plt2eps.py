#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-03-15 10:08'
from PIL import Image #将模块导入
import os

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist

dataset_dir= os.getcwd()
image_filenames = get_img_file(dataset_dir)
print(image_filenames)

for i in image_filenames:
    fname = i.strip('.jpg')
    img=Image.open(i) #打开图片
    final_name = fname+".eps"
    img.save(final_name,dpi=300) #保存图片
    print(final_name+"保存完成！")

