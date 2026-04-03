import torch
print("\n"*5)

softmax = torch.nn.Softmax(dim=-1)
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
output_tensor = softmax(input_tensor)  
sum_across_last_dim = output_tensor[0].sum()
print(f"Input tensor:\n{input_tensor}")
print(f"Softmax output tensor:\n{output_tensor}")
print(f"Softmax output tensor last dimension:\n{output_tensor[0]}")
print(f"Sum of softmax output across the last dimension: {sum_across_last_dim}") 

# add up tensor values across the last dimension