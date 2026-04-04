import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int =1 , output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
    def criterion(self, y_pred, y_true):
        loss_fn = nn.BCELoss()
        return loss_fn(y_pred, y_true)
    
    def fit(self, X, y_true, learning_rate=0.01, epochs=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = self.criterion
        
        for epoch in range(epochs):
            self.train()
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y_true)
            
            if loss.item() < 0.01:
                print(f"Epoch {epoch}: loss: {loss.item():.6f} - stopping training.")
                break
            
            if (epoch % 10 == 0):
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()