import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LinearRegressionDS(nn.Module):
    """
    Simple linear regression model: y = X * w + b
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        """
        Initialize linear regression model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super(LinearRegressionDS, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = X * w + b
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted output tensor of shape (batch_size, output_dim)
        """
        return self.linear(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (same as forward but in evaluation mode).
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted output tensor
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get weight and bias parameters.
        
        Returns:
            Tuple of (weight, bias) tensors
        """
        weight = self.linear.weight.data
        bias = self.linear.bias.data
        return weight, bias
    
    def set_parameters(self, weight: torch.Tensor, bias: torch.Tensor):
        """
        Set weight and bias parameters.
        
        Args:
            weight: Weight tensor
            bias: Bias tensor
        """
        self.linear.weight.data = weight
        self.linear.bias.data = bias


def generate_synthetic_data(
    n_samples: int = 100,
    input_dim: int = 1,
    output_dim: int = 1,
    noise_std: float = 0.1,
    seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic linear regression data.
    
    Args:
        n_samples: Number of samples
        input_dim: Number of input features
        output_dim: Number of output features
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y_true, true_weight, true_bias)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random input data
    X = torch.randn(n_samples, input_dim)
    
    # True parameters
    true_weight = torch.randn(input_dim, output_dim) * 2.0
    true_bias = torch.randn(output_dim) * 1.0
    
    # Generate target with noise
    y_true = X @ true_weight + true_bias + noise_std * torch.randn(n_samples, output_dim)
    
    return X, y_true, true_weight, true_bias


def train_linear_regression(
    model: LinearRegressionDS,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 1000,
    print_every: int = 100
) -> Tuple[list, list]:
    """
    Train linear regression model.
    
    Args:
        model: Linear regression model
        X: Input features
        y: Target values
        learning_rate: Learning rate for optimizer
        n_epochs: Number of training epochs
        print_every: Print loss every n epochs
        
    Returns:
        Tuple of (loss_history, weight_history)
    """
    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()
    
    # Optimizer: Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    weight_history = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = criterion(y_pred, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store history
        loss_history.append(loss.item())
        weight, bias = model.get_parameters()
        weight_history.append(weight.clone().detach())
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}')
    
    return loss_history, weight_history


def plot_training_results(
    X: torch.Tensor,
    y_true: torch.Tensor,
    model: LinearRegressionDS,
    loss_history: list,
    true_weight: Optional[torch.Tensor] = None,
    true_bias: Optional[torch.Tensor] = None
):
    """
    Plot training results.
    
    Args:
        X: Input features
        y_true: True target values
        model: Trained model
        loss_history: List of loss values during training
        true_weight: True weight parameters (if known)
        true_bias: True bias parameters (if known)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Data and regression line
    axes[0].scatter(X.numpy(), y_true.numpy(), alpha=0.6, label='True data')
    
    # Generate predictions for plotting
    X_plot = torch.linspace(X.min(), X.max(), 100).unsqueeze(1)
    y_pred = model.predict(X_plot)
    axes[0].plot(X_plot.numpy(), y_pred.numpy(), 'r-', linewidth=2, label='Regression line')
    
    if true_weight is not None and true_bias is not None:
        # Plot true regression line
        y_true_line = X_plot @ true_weight + true_bias
        axes[0].plot(X_plot.numpy(), y_true_line.numpy(), 'g--', linewidth=2, label='True line')
    
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('Linear Regression Fit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curve
    axes[1].plot(loss_history)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Plot 3: Predictions vs True values
    y_pred_all = model.predict(X)
    axes[2].scatter(y_true.numpy(), y_pred_all.detach().numpy(), alpha=0.6)
    axes[2].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
    axes[2].set_xlabel('True y')
    axes[2].set_ylabel('Predicted y')
    axes[2].set_title('Predictions vs True Values')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate linear regression.
    """
    print("=" * 60)
    print("Linear Regression Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y_true, true_weight, true_bias = generate_synthetic_data(
        n_samples=100,
        input_dim=1,
        output_dim=1,
        noise_std=0.2,
        seed=42
    )
    
    print(f"   Data shape: X={X.shape}, y={y_true.shape}")
    print(f"   True weight: {true_weight.item():.4f}, True bias: {true_bias.item():.4f}")
    
    # Create model
    print("\n2. Creating linear regression model...")
    model = LinearRegressionDS(input_dim=1, output_dim=1)
    initial_weight, initial_bias = model.get_parameters()
    print(f"   Initial weight: {initial_weight.item():.4f}, Initial bias: {initial_bias.item():.4f}")
    
    # Train model
    print("\n3. Training model...")
    loss_history, weight_history = train_linear_regression(
        model=model,
        X=X,
        y=y_true,
        learning_rate=0.01,
        n_epochs=1000,
        print_every=250
    )
    
    # Get final parameters
    final_weight, final_bias = model.get_parameters()
    print(f"\n4. Training completed!")
    print(f"   Final weight: {final_weight.item():.4f}, Final bias: {final_bias.item():.4f}")
    print(f"   True weight:  {true_weight.item():.4f}, True bias:  {true_bias.item():.4f}")
    print(f"   Weight error: {abs(final_weight.item() - true_weight.item()):.6f}")
    print(f"   Bias error:   {abs(final_bias.item() - true_bias.item()):.6f}")
    print(f"   Final loss:   {loss_history[-1]:.6f}")
    
    # Plot results
    print("\n5. Plotting results...")
    plot_training_results(
        X=X,
        y_true=y_true,
        model=model,
        loss_history=loss_history,
        true_weight=true_weight,
        true_bias=true_bias
    )
    
    # Make predictions on new data
    print("\n6. Making predictions on new data...")
    X_new = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
    y_pred_new = model.predict(X_new)
    
    print("   New predictions:")
    for i in range(len(X_new)):
        print(f"   X={X_new[i].item():.2f} -> y_pred={y_pred_new[i].item():.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()