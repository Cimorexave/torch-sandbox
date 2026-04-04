import torch
import sys
sys.path.insert(0, '.')
from lin_reg_ds import LinearRegressionDS, generate_synthetic_data, train_linear_regression

def test_basic_functionality():
    """Test basic functionality of linear regression model."""
    print("Testing LinearRegressionDS basic functionality...")
    
    # 1. Test model creation
    model = LinearRegressionDS(input_dim=2, output_dim=1)
    print(f"[OK] Model created: {model}")
    
    # 2. Test forward pass
    X = torch.randn(5, 2)
    y_pred = model(X)
    print(f"[OK] Forward pass works: input shape {X.shape}, output shape {y_pred.shape}")
    
    # 3. Test parameter access
    weight, bias = model.get_parameters()
    print(f"[OK] Parameters accessible: weight shape {weight.shape}, bias shape {bias.shape}")
    
    # 4. Test parameter setting
    new_weight = torch.tensor([[1.5], [2.0]])
    new_bias = torch.tensor([0.5])
    model.set_parameters(new_weight, new_bias)
    weight, bias = model.get_parameters()
    print(f"[OK] Parameters settable: weight={weight.tolist()}, bias={bias.tolist()}")
    
    return True

def test_training():
    """Test training functionality."""
    print("\nTesting training functionality...")
    
    # Generate synthetic data
    X, y_true, true_weight, true_bias = generate_synthetic_data(
        n_samples=50,
        input_dim=1,
        output_dim=1,
        noise_std=0.1,
        seed=42
    )
    
    print(f"[OK] Data generated: {X.shape[0]} samples")
    
    # Create and train model
    model = LinearRegressionDS(input_dim=1, output_dim=1)
    initial_weight, initial_bias = model.get_parameters()
    
    loss_history, weight_history = train_linear_regression(
        model=model,
        X=X,
        y=y_true,
        learning_rate=0.01,
        n_epochs=100,
        print_every=50
    )
    
    final_weight, final_bias = model.get_parameters()
    
    print(f"[OK] Training completed: {len(loss_history)} epochs")
    print(f"  Initial loss: {loss_history[0]:.4f}")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    print(f"  Loss reduced by: {loss_history[0] - loss_history[-1]:.4f}")
    
    # Check if loss decreased
    if loss_history[-1] < loss_history[0]:
        print("[OK] Loss decreased during training")
    else:
        print("[FAIL] Loss did not decrease")
        
    return loss_history[-1] < loss_history[0]

def test_predictions():
    """Test prediction functionality."""
    print("\nTesting prediction functionality...")
    
    # Create a simple model with known parameters
    model = LinearRegressionDS(input_dim=1, output_dim=1)
    model.set_parameters(torch.tensor([[2.0]]), torch.tensor([1.0]))
    
    # Test predictions
    X_test = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y_pred = model.predict(X_test)
    
    print("[OK] Predictions:")
    for i in range(len(X_test)):
        x_val = X_test[i].item()
        y_val = y_pred[i].item()
        expected = 2.0 * x_val + 1.0
        error = abs(y_val - expected)
        print(f"  X={x_val:.1f}: predicted={y_val:.4f}, expected={expected:.4f}, error={error:.6f}")
        
        if error > 0.001:
            print(f"  [FAIL] Prediction error too large for X={x_val}")
            return False
    
    print("[OK] All predictions match expected values (within tolerance)")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Linear Regression Implementation")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    try:
        all_passed &= test_basic_functionality()
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_training()
    except Exception as e:
        print(f"[FAIL] Training test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_predictions()
    except Exception as e:
        print(f"[FAIL] Prediction test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] All tests passed!")
    else:
        print("[FAIL] Some tests failed")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)