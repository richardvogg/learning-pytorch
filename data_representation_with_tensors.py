# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:58:55 2021

@author: Richard
"""

import imageio
import torch
import os #working directory and path to images
import numpy as np


#%%

#read an image

img_arr = imageio.imread("Pictures/web.png")

img_arr.shape

#%%


img = torch.from_numpy(img_arr)

out = img.permute(2, 0, 1)

#out uses the same underlying storage as img and only plays with the size and 
#strid information at the tensor level.
#Changing a pixel in img will lead to a change in out!

#%%

#reading multiple images and adding them to a batch

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype = torch.uint8)

data_dir = "data/image-cats"
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == ".png"]

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir,filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]
    batch[i] = img_t


#%%

#tabular data

import csv

path = "C:/Richard/R and Python/Datasets/diabetes.csv"

test_df = np.loadtxt(path,
                     dtype = np.float32, delimiter = ";", 
                     skiprows = 1) # skiprows to remove column headers

col_list = next(csv.reader(open(path), delimiter = ";"))

test_torch_df = torch.from_numpy(test_df)

test_torch_df.shape, test_torch_df.dtype