import torch
print("\n"*5)


D_in = 1
D_out = 1

X = torch.randn(10, D_in)
y_real = 2 * X + 1 + 0.1 * torch.randn(10, D_out)


linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)
print(f"Initial weight: {linear_layer.weight}, Initial bias: {linear_layer.bias}")

# linear layer is the forward pass
y_hat_nn = linear_layer(X)
print(f"Predicted y_hat from nn.Linear: {y_hat_nn[:10]}") 
print(f"Real y: {y_real[:10]}") 



import torch
print("\n"*5)

# ------------ DATA
N=10
D_in = 1
D_out = 1

X = torch.randn(N, D_in)

true_w = torch.tensor([[2.0]],)
true_b = torch.tensor(1.0)
true_y = X @ true_w + true_b + 0.1 * torch.randn(N, D_out)


#------------ BRAIN
learning_rate , epochs = 0.01, 1000
linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)

for epoch in range(epochs):
    # forward pass
    y_prediction = linear_layer(X)
    # y_prediction = torch.nn.Linear(in_features=D_in, out_features=D_out)(X) 
    loss = ((y_prediction - true_y)**2).mean()
    
    if loss.item() < 0.01:
        print(f"Epoch {epoch}: loss: {loss.item():.6f} - stopping training.")
        break
    
    if (epoch % 10 == 0):
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, w = {linear_layer.weight}, b = {linear_layer.bias}")
    
    loss.backward()
    
    with torch.no_grad():
        # w -= 0.01 * w.grad
        # b -= 0.01 * b.grad
        linear_layer.weight  = learning_rate * linear_layer.weight.grad
        linear_layer.bias = learning_rate * linear_layer.bias.grad
        
        # w.grad.zero_()
        # b.grad.zero_()
        linear_layer.weight.grad.zero_()
        linear_layer.bias.grad.zero_()