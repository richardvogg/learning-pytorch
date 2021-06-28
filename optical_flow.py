# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:29:00 2021

@author: Richard
"""

import cv2
import numpy as np

path = "C:/Richard/R and Python/Deep Learning/Video Labeling/"

#%%

frame1 = cv2.imread(path + "frame0.jpg")
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#for i in range(0, 700, 10):
for i in 240:
    print(i)

    frame2 = cv2.imread(path + "frame" + str(i+10) + ".jpg")
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    hsv[:,:,1] = 255

    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, flow = None, 
                                        pyr_scale = 0.5, levels = 3, 
                                        winsize = 15, iterations = 3,
                                        poly_n = 5, poly_sigma = 1.2, flags =  0)
    
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    #angle of movement
    hsv[:,:, 0] = ang * 180 / np.pi / 2
    #strength of movement
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imwrite(path + "flow" + str(i) + ".jpg", rgb)
    
    frame1_gray = frame2_gray
    
#%%



cv2.imwrite(path + "test.png", rgb[:,:,2])

#%%

#histogram

import matplotlib.pyplot as plt


fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

vec = hsv[:,:,2].flatten()

axs[0].hist(hsv[:,:,0].flatten(), bins=255)
axs[1].hist(vec[vec>10], bins=255)