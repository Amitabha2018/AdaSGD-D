#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-03-15 16:04'

import glob
import fitz  # 导入本模块需安装pymupdf库
import os

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist

dataset_dir = os.getcwd()
image_filenames = get_img_file(dataset_dir)


def img2pdf(img_path):
    pdf_path = img_path.replace("jpg","pdf")
    print("pdf文件路径为:",pdf_path)
    doc = fitz.open()
    imgdoc = fitz.open(img_path,)                 # 打开图片
    pdfbytes = imgdoc.convertToPDF()        # 使用图片创建单页的 PDF
    imgpdf = fitz.open("pdf", pdfbytes)
    doc.insertPDF(imgpdf)                   # 将当前页插入文档
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        doc.save(pdf_path)                   # 保存pdf文件
        doc.close()
    doc.save(pdf_path)
    print(pdf_path+'已保存!')

for i in image_filenames:
    img2pdf(i)
