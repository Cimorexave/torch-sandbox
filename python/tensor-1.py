import torch

from torch import Tensor

data = [[1,2,3,], [4,5,6]]
t_data = torch.tensor(data)
# print(t_data)
# print(t_data.dtype)

shape = (2,3)
# rand_tensor = torch.rand(shape)
# print(rand_tensor)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
# print(zeros)
# print(ones)

likeness_zero = torch.rand_like(zeros, dtype=zeros.dtype)
# print(f"templaete zeros: {zeros}, \nlikeness zeros: {likeness_zero}")

# properties
tensor = torch.rand(3,4)
# print(f"shape: {tensor.shape}, \ndtype: {tensor.dtype}, \ndevice: {tensor.device}")

t_data = torch.tensor(data, dtype=torch.int16)
param_data = torch.tensor(data, dtype=torch.float32, device="cpu", requires_grad=True)

