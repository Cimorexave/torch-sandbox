import torch
print("\n"*5)

norm = torch.nn.LayerNorm(normalized_shape=3)
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
normalized_tensor = norm(input_tensor)
print(f"Input tensor:\n{input_tensor}")
print(f"Normalized tensor:\n{normalized_tensor}")