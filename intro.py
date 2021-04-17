# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:46:32 2021

@author: Richard

from: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
"""

import torch
import numpy as np
import torchvision

#%%

# create a tensor from a list

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#%%
# from numpy array

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#%%
#from another tensor

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

test = torch.ones(2,3)

test2 = torch.ones((2,3,))

#%%

#Attributes: shape, data type, device of a tensor

print(f"Shape:  {test.shape} \n")
print(f"Data type: {test.dtype} \n")
print(f"Device tensor is stored on: {test.device}")

#%% 

#slicing - as in numpy

tensor = torch.ones(4, 4)
tensor[:2,1] = 64
print(tensor)


#%%

#concatenating
#dim = 0: concat along rows
#dim = 1: concat along columns

large_object = torch.cat([tensor,tensor],dim = 0)

#%%

#point-wise multiplication

tensor * tensor

#or

tensor.mul(tensor)

#%%

# matrix multiplication

tensor.matmul(tensor.T)

# or

tensor @ tensor.T


#%%

#in-place operations (use is discouraged)

tensor.add_(2)
print(tensor)


