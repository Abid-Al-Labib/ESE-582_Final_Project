import numpy as np
import torch

values = np.array([[80.0, 37.0], [90.0, 0.0]], dtype=np.float32)
mask = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
print(values.shape)             # two hours, two features
print(values[:, 0])            # heart rate at both hours
print(values * mask)           # element-by-element multiplication
batch = torch.from_numpy(values).unsqueeze(0)
print(batch.shape) 
print(torch)