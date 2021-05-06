# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:29:05 2021

@author: Richard
"""

from torchvision import models
from torchvision import transforms
import torch

#%%

#all available models
dir(models)

#%%

resnet = models.resnet101(pretrained = True)

#%%

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    ])

#%%

from PIL import Image

img = Image.open("C:/Richard/Fotos/2020/07_Juli/Valpo Neue Kamera/DSC_0025.jpg")

img_t = preprocess(img)

#%%

batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()

out = resnet(batch_t)

#Get top 1 predictions

_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim = 1)[0] * 100

print(index[0], percentage[index[0]].item())

# Get top n prodictions

_, indices = torch.sort(out, descending = True)

print([(idx, percentage[idx].item()) for idx in indices[0][:5]])

