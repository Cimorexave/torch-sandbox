"""
Example usage of the LinearRegressionDS model.
This demonstrates how to use the linear regression implementation.
"""

import torch
from lin_reg_ds import LinearRegressionDS, generate_synthetic_data, train_linear_regression

def simple_example():
    """Simple example of using linear regression."""
    print("=" * 60)
    print("Linear Regression Example Usage")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y_true, true_weight, true_bias = generate_synthetic_data(
        n_samples=100,
        input_dim=1,
        output_dim=1,
        noise_std=0.2,
        seed=42
    )
    
    print(f"   Generated {X.shape[0]} samples")
    print(f"   True parameters: weight={true_weight.item():.4f}, bias={true_bias.item():.4f}")
    
    # 2. Create model
    print("\n2. Creating linear regression model...")
    model = LinearRegressionDS(input_dim=1, output_dim=1)
    initial_weight, initial_bias = model.get_parameters()
    print(f"   Initial parameters: weight={initial_weight.item():.4f}, bias={initial_bias.item():.4f}")
    
    # 3. Train model
    print("\n3. Training model...")
    loss_history, _ = train_linear_regression(
        model=model,
        X=X,
        y=y_true,
        learning_rate=0.01,
        n_epochs=500,
        print_every=100
    )
    
    # 4. Check results
    print("\n4. Training results:")
    final_weight, final_bias = model.get_parameters()
    print(f"   Final parameters: weight={final_weight.item():.4f}, bias={final_bias.item():.4f}")
    print(f"   True parameters:  weight={true_weight.item():.4f}, bias={true_bias.item():.4f}")
    print(f"   Weight error: {abs(final_weight.item() - true_weight.item()):.6f}")
    print(f"   Bias error:   {abs(final_bias.item() - true_bias.item()):.6f}")
    print(f"   Initial loss: {loss_history[0]:.6f}")
    print(f"   Final loss:   {loss_history[-1]:.6f}")
    
    # 5. Make predictions
    print("\n5. Making predictions...")
    X_test = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
    y_pred = model.predict(X_test)
    
    print("   Test predictions:")
    for i in range(len(X_test)):
        print(f"   X={X_test[i].item():.2f} -> y_pred={y_pred[i].item():.4f}")
    
    # 6. Manual calculation verification
    print("\n6. Manual verification:")
    print(f"   For X=1.0: y = {final_weight.item():.4f} * 1.0 + {final_bias.item():.4f} = {final_weight.item() + final_bias.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

def multi_feature_example():
    """Example with multiple input features."""
    print("\n" + "=" * 60)
    print("Multi-Feature Linear Regression Example")
    print("=" * 60)
    
    # Generate data with 3 input features
    torch.manual_seed(123)
    n_samples = 200
    input_dim = 3
    output_dim = 1
    
    # True parameters
    true_weight = torch.tensor([[1.5], [-0.8], [2.2]])  # 3x1
    true_bias = torch.tensor([0.5])
    
    # Generate random input
    X = torch.randn(n_samples, input_dim)
    
    # Generate target with noise
    y_true = X @ true_weight + true_bias + 0.1 * torch.randn(n_samples, output_dim)
    
    print(f"\nGenerated data: {n_samples} samples, {input_dim} features")
    print(f"True weights: {true_weight.squeeze().tolist()}")
    print(f"True bias: {true_bias.item():.4f}")
    
    # Create and train model
    model = LinearRegressionDS(input_dim=input_dim, output_dim=output_dim)
    
    loss_history, _ = train_linear_regression(
        model=model,
        X=X,
        y=y_true,
        learning_rate=0.005,
        n_epochs=1000,
        print_every=250
    )
    
    # Get learned parameters
    learned_weight, learned_bias = model.get_parameters()
    
    print(f"\nLearned weights: {learned_weight.squeeze().tolist()}")
    print(f"Learned bias: {learned_bias.item():.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    # Test prediction
    X_test = torch.tensor([[1.0, 0.5, -0.3], [0.0, 1.0, 0.5]])
    y_pred = model.predict(X_test)
    
    print(f"\nTest predictions:")
    for i in range(len(X_test)):
        print(f"  Sample {i+1}: {X_test[i].tolist()} -> {y_pred[i].item():.4f}")
    
    print("\n" + "=" * 60)
    print("Multi-feature example completed!")
    print("=" * 60)

if __name__ == "__main__":
    simple_example()
    multi_feature_example()
    
    print("\n\nKey features of the LinearRegressionDS implementation:")
    print("1. Simple PyTorch nn.Module subclass")
    print("2. Built-in training function with SGD optimizer")
    print("3. Synthetic data generation utility")
    print("4. Parameter getter/setter methods")
    print("5. Prediction method with eval mode")
    print("6. Support for multiple input/output features")