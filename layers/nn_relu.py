import torch
print("\n"*5)

tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
relu = torch.nn.ReLU()
gelu = torch.nn.GELU()
activated_data_1 = relu(tensor)
activated_data_2 = gelu(tensor)
print(f"Input tensor: {tensor}")
print(f"Activated tensor (ReLU): {activated_data_1}")
print(f"Activated tensor (GELU): {activated_data_2}")
