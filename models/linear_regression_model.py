import torch
print("\n"*5)

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_features, out_features): # constructor method to initialize the linear layer with specified input and output features
        super(LinearRegressionModel, self).__init__()
        # define layers
        self.linear_layer = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_layer(x)
    
    # def loss(self, y_pred, y_true):
    #     return ((y_pred - y_true)**2).mean()

    def loss(self, y_pred, y_true):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        return loss_fn(y_pred, y_true)
    
    def train(self, X, y_true, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            # forward pass
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y_true)
            
            if loss.item() < 0.01:
                print(f"Epoch {epoch}: loss: {loss.item():.6f} - stopping training.")
                break
            
            if (epoch % 10 == 0):
                print(f"Epoch {epoch}: loss = {loss.item():.6f}, w = {self.linear_layer.weight}, b = {self.linear_layer.bias}")
            
            # loss.backward()
            
            # with torch.no_grad():
            #     self.linear_layer.weight -= learning_rate * self.linear_layer.weight.grad
            #     self.linear_layer.bias -= learning_rate * self.linear_layer.bias.grad
                
            #     self.linear_layer.weight.grad.zero_()
            #     self.linear_layer.bias.grad.zero_()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# Example usage

X = torch.randn(10, 1)
true_w = torch.tensor([[2.0]],)
true_b = torch.tensor(1.0)
true_y = X @ true_w + true_b + 0.1 * torch.randn(10, 1)

model = LinearRegressionModel(in_features=1, out_features=1)
print(f"model: {model}")

# initialized the model with layers which is one lineary layer here.
# we call it to train with data
model.train(X, true_y)
print(f"Trained weight: {model.linear_layer.weight}, Trained bias: {model.linear_layer.bias}")  


# model.eval()  # set the model to evaluation mode
# y_pred = model.forward(X)
