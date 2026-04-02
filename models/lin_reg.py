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


# ------------ BRAIN
w = torch.rand_like(true_w, requires_grad=True)
b = torch.rand_like(true_b, requires_grad=True)
print(f"w: {w}, \nb: {b}\n")

def forward(X, W, B):
    """" X: input data, shape (N, D_in) """
    return X @ W + B

# prediction
def model(X, W: torch.Tensor, B: torch.Tensor, y_real: torch.Tensor, n_epochs: int = 100):
    y_prediction = forward(X, W, B)
    loss = ((y_prediction - y_real)**2).mean()
    
    if loss.item() < 0.01 or n_epochs == 0:
        print(f"Epoch {n_epochs} loss: {loss.item():.6f} - stopping training.")
        return W_new, B_new, loss.item(), True
    
    if (n_epochs % 10 == 0):
        print(f"Epoch {n_epochs}: loss = {loss.item():.6f}, W = {W}, B = {B}")

    loss.backward()
    # what is the 0.01? it's the learning rate, it controls how big of a step we take in the direction of the negative gradient. if it's too small, training will be slow, if it's too big, we might overshoot the minimum and diverge.
    learning_rate = 0.01
    W_new , B_new = torch.zeros_like(W, requires_grad=True), torch.zeros_like(B, requires_grad=True)    

    W_new = W - learning_rate * W.grad
    B_new = B - learning_rate * B.grad

    model(X, W_new, B_new, y_real, n_epochs=n_epochs-1)
    
final_w , final_b, final_loss, converged = model(X, w, b, true_y)
print(f"final_w: {final_w}, final_b: {final_b}, final_loss: {final_loss}, converged: {converged}")

# y_hat = forward(X)
# print(f" prediction y_hat: {y_hat}")
# print(f" real y: {true_y}")


# loss
# loss = ((y_hat - true_y)**2).mean()
# loss = torch.nn.functional.mse_loss(y_hat, true_y)
# print(f"loss: {loss}")

# loss.backward()
# print(f"dL/dw: {w.grad}, \ndL/db: {b.grad}")

# w_new = w -w.grad * 0.01
# b_new = b -b.grad * 0.01

# print(f"w_new: {w_new}, \nb_new: {b_new}")
# forward(X, w_new, b_new)
