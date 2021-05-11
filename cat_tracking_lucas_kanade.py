# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:13:02 2021

@author: Richard
"""

#from https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/

import cv2
import numpy as np

#%%

cap = cv2.VideoCapture("C:/Richard/Smartphone/Bilder3/Test.mp4")

  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (10, 10),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
  
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
  
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,
                        cv2.COLOR_BGR2GRAY)

#Coordinates of cat heads
p0 = np.array([[[620,270]], [[615, 325]]], dtype = "float32")

#%%

mask = np.zeros_like(old_frame)

while(1):
      
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
  
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           p0, None,
                                           **lk_params)
  
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
  
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, 
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                        color[i].tolist(), 2)
          
        frame = cv2.circle(frame, (int(a), int(b)), 5,
                           color[i].tolist(), -1)
          
    img = cv2.add(frame, mask)
  
    cv2.imshow('frame', img)
      
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  
    # Updating Previous frame and points 
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
  
cv2.destroyAllWindows()
cap.release()