import torch

from torch import Tensor

data = [[1,2,3,], [4,5,6]]
t_data = torch.tensor(data)
print(t_data)
print(t_data.dtype)