import torch
print("\n"*5)

# ------------ DATA
N=10
D_in = 1
D_out = 1

X = torch.randn(N, D_in)
print(f"X: {X}")

true_w = torch.tensor([[2.0]],)
true_b = torch.tensor(1.0)
true_y = X @ true_w + true_b + 0.1 * torch.randn(N, D_out)


#------------ BRAIN
w, b = torch.rand_like(true_w, requires_grad=True), torch.rand_like(true_b, requires_grad=True)
epochs = 1000
for epoch in range(epochs):
    # forward pass
    y_prediction = X @ w + b
    loss = ((y_prediction - true_y)**2).mean()
    
    if loss.item() < 0.01:
        print(f"Epoch {epoch}: loss: {loss.item():.6f} - stopping training.")
        break
    
    if (epoch % 10 == 0):
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, w = {w}, b = {b}")
    
    loss.backward()
    
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
        
        w.grad.zero_()
        b.grad.zero_()