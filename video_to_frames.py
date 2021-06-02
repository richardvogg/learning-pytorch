# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:07:15 2021

@author: Richard
"""

import cv2
path = "C:/Richard/R and Python/Deep Learning/Video Labeling/"

cap = cv2.VideoCapture(path + 'VID_20200501_112456small.mp4')

#Total number of frames
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)

count = 0

#%%

while count <= frame_count:
  cap.set(cv2.CAP_PROP_POS_FRAMES, count)
  success,image = cap.read()
  cv2.imwrite(path + "frame%d.jpg" % count, image)     # save frame as JPEG file
  #if cv2.waitKey(10) == 27:                     # exit if Escape is hit
  #    break
  count += 10