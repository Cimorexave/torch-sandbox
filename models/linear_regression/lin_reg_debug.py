import torch

print("\n"*2)
print("=" * 60)
print("DEBUG VERSION - Analyzing lin_reg.py issues")
print("=" * 60)

# ------------ DATA
N=10
D_in = 1
D_out = 1

X = torch.randn(N, D_in)
print(f"X: {X}")

true_w = torch.tensor([[2.0]],)
true_b = torch.tensor(1.0)
true_y = X @ true_w + true_b + 0.1 * torch.randn(N, D_out)


# ------------ BRAIN
w = torch.rand_like(true_w, requires_grad=True)
b = torch.rand_like(true_b, requires_grad=True)
print(f"Initial w: {w}, requires_grad={w.requires_grad}, is_leaf={w.is_leaf}")
print(f"Initial b: {b}, requires_grad={b.requires_grad}, is_leaf={b.is_leaf}\n")

def forward(X, W, B):
    """" X: input data, shape (N, D_in) """
    return X @ W + B

# prediction
def model(X, W: torch.Tensor, B: torch.Tensor, y_real: torch.Tensor, n_epochs: int = 100, depth=0):
    print(f"\n--- Model call depth={depth}, n_epochs={n_epochs} ---")
    print(f"W: {W}, requires_grad={W.requires_grad}, is_leaf={W.is_leaf}, grad={W.grad}")
    print(f"B: {B}, requires_grad={B.requires_grad}, is_leaf={B.is_leaf}, grad={B.grad}")
    
    y_prediction = forward(X, W, B)
    loss = ((y_prediction - y_real)**2).mean()
    
    if loss.item() < 0.01:
        print(f"loss: {loss.item():.6f} - stopping training; reached loss < 0.01")
        print(f"final W: {W}, final B: {B}")
        return W, B, loss.item(), True
    
    if (n_epochs % 10 == 0):
        print(f"Epoch {100-n_epochs}: loss = {loss.item():.6f}, W = {W}, B = {B}")
    
    # Check gradients before backward
    print(f"Before backward - W.grad: {W.grad}, B.grad: {B.grad}")
    
    loss.backward()
    
    # Check gradients after backward
    print(f"After backward - W.grad: {W.grad}, B.grad: {B.grad}")
    
    # what is the 0.01? it's the learning rate
    learning_rate = 0.01
    
    # Check if gradients exist
    if B.grad is None:
        print("ERROR: B.grad is None!")
        # Try to create new leaf tensors
        B_new = torch.tensor(B.item() - learning_rate * 0.0, requires_grad=True)
    else:
        B_new = B - learning_rate * B.grad
    
    if W.grad is None:
        print("ERROR: W.grad is None!")
        # Try to create new leaf tensors
        W_new = torch.tensor(W.item() - learning_rate * 0.0, requires_grad=True)
    else:
        W_new = W - learning_rate * W.grad
    
    print(f"W_new: {W_new}, requires_grad={W_new.requires_grad}, is_leaf={W_new.is_leaf}")
    print(f"B_new: {B_new}, requires_grad={B_new.requires_grad}, is_leaf={B_new.is_leaf}")
    
    # Zero gradients for next iteration
    if W.grad is not None:
        W.grad.zero_()
    if B.grad is not None:
        B.grad.zero_()
    
    # Recursive call with decremented epochs
    return model(X, W_new, B_new, y_real, n_epochs-1, depth+1)

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

try:
    final_w, final_b, final_loss, converged = model(X, w, b, true_y, n_epochs=100)
    print(f"\nfinal_w: {final_w}, final_b: {final_b}, final_loss: {final_loss}, converged: {converged}")
except RecursionError as e:
    print(f"\nRecursionError: {e}")
    print("The function recurses too deep - needs proper loop instead of recursion")
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)