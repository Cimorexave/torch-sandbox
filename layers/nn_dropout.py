import torch
print("\n"*5)

dropout = torch.nn.Dropout(p=0.5)
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

dropout.train()  # set dropout to training mode
output_tensor_train = dropout(input_tensor)  # apply dropout to input tensor in training mode 

dropout.eval()  # set dropout to evaluation mode
output_tensor_eval = dropout(input_tensor)  # apply dropout to input tensor in evaluation mode
print(f"Input tensor:\n{input_tensor}")

print(f"Output tensor (training mode):\n{output_tensor_train}")
print(f"Output tensor (evaluation mode):\n{output_tensor_eval}")