import torch
import torch.nn as nn

class DeepWeatherEvaluator(nn.Module):
    def __init__(self, in_dimension = 5, out_dimension = 1):
        super(DeepWeatherEvaluator, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(in_dimension, in_dimension * 8),
            nn.ReLU(),
            nn.Linear(in_dimension * 8, out_dimension),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(in_dimension * 4, out_dimension),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X)
            return (y_pred > 0.5).float()  # convert probabilities to binary predictions
    
    def fit(self, X, y_true, learning_rate=0.01, epochs=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.BCELoss()  # binary cross-entropy loss
        
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