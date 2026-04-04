import torch
import torch.nn as nn

class DeepWeatherEvaluator(nn.Module):
    def __int__(self, in_dimension = 5, out_dimension = 1):
        super.__init__(DeepWeatherEvaluator, self).__init__()
        torch.nn.Sequential(
            nn.Linear(in_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def criterion(self, y_pred, y_true):
        # cross entropy loss for binary classification
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
            
            if (epoch % 100 == 0):
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()