# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:26:37 2021

@author: Richard
"""

import numpy as np
import cv2

#%%

img = np.zeros((512,512,3), np.uint8)

cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
cv2.circle(img,(400, 50), 30, (255,120,110), 2)

cv2.putText(img, "Hello", (300, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,150,0), 1)

while(1):
    cv2.imshow("Image", img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()