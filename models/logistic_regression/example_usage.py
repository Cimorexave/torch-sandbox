import torch
from lgr import LogisticRegressionModel
print("\n"*5)

if __name__ == "__main__":
    print("=" * 60)
    print("Single-Feature Logistic Regression Example")
    print("=" * 60)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 100
    input_dim = 1
    output_dim = 1

    # True parameters
    true_weight = torch.tensor([[2.0]])  # 1x1
    true_bias = torch.tensor([0.5])

    # Generate random input
    X = torch.randn(n_samples, input_dim)

    # Generate target with noise
    y_true = (X @ true_weight + true_bias > 0).float() + 0.1 * torch.randn(n_samples, output_dim)
    y_true = (y_true > 0.5).float()

    print(f"\nGenerated data: {n_samples} samples, {input_dim} feature")
    print(f"True weight: {true_weight.item():.4f}")
    print(f"True bias: {true_bias.item():.4f}")

    # Create and train model

    model = LogisticRegressionModel(input_dim=input_dim, output_dim=output_dim)
    model.fit(X, y_true, learning_rate=0.01, epochs=1000)
    # Test prediction
    X_test = torch.tensor([[1.0], [-1.0], [0.0]])
    y_pred = model.predict(X_test)
    print(f"\nTest predictions:")
    for i in range(len(X_test)):
        print(f"  Sample {i+1}: {X_test[i].tolist()} -> {y_pred[i].item():.4f}")

    print("\n" + "=" * 60)
    print("Single-feature example completed!")
    print("=" * 60)


