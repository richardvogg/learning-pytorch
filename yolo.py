# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:15:30 2021

@author: Richard
"""

import torch
from torchvision import transforms
from PIL import Image

#%%

#https://pytorch.org/hub/ultralytics_yolov5/
# this is the smallest version (s) - for better models check m, l and xl

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#%%

dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batched list of images

# Inference
results = model(imgs)

#%%

# with own images

img = "C:/Richard/Fotos/2020/07_Juli/Valpo Neue Kamera/DSC_0025.jpg"

mews = model(img)

#%%

#with PIL


img = Image.open("C:/Richard/Fotos/2020/07_Juli/Valpo Neue Kamera/DSC_0025.jpg")


imgs = [img]

results2 = model(imgs, size = 640)

#%%

results2.print()
#results2.show()
results2.save("C:/Richard/Fotos")  # or .show()


#%%

imgs = []

for i in range(1,13):
    img = Image.open("C:/Richard/Fotos/Gatos/"+ str(i) +".jpg")
    imgs.append(img)
    
    
results3 = model(imgs, size = 640)

#%%

results3.save("C:/Richard/Fotos")


#%%

import cv2

cap = cv2.VideoCapture("C:/Richard/Fotos/Ohne Titel.mp4")

#%%

imgs = []

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  if ret == True:
    # Display the resulting frame
    
    #results = model(frame)
    imgs.append(frame)
    cv2.imshow('Frame', frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#%%

# YOLO on all images (takes some time!)

model_img = []

for img in imgs:
    results4 = model(img)
    model_img.append(results4)
    
# Check frames

model_img[600].print()
model_img[600].show()
model_img[700].pandas().xyxy[0]