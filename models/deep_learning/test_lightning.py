"""
Quick test to verify the PyTorch Lightning implementation works correctly.
"""
import torch
import pytorch_lightning as pl
from dwe_lightning import DeepWeatherEvaluatorLightning


def test_model_creation():
    """Test that the model can be created and forward pass works."""
    print("Testing model creation...")
    
    # Create model with default parameters
    model = DeepWeatherEvaluatorLightning(
        input_dim=5,
        output_dim=1,
        hidden_dims=[40, 20],
        dropout_rate=0.3,
        use_batch_norm=True,
        activation="relu",
        learning_rate=0.001,
    )
    
    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 5)
    output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be in [0, 1] range"
    
    print("  [OK] Model creation test passed")
    return model


def test_training_step():
    """Test that training step works correctly."""
    print("\nTesting training step...")
    
    model = DeepWeatherEvaluatorLightning(input_dim=5)
    
    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 5)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Test training step
    loss = model.training_step((x, y), batch_idx=0)
    
    print(f"  Loss value: {loss.item():.4f}")
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients"
    
    print("  [OK] Training step test passed")
    return loss


def test_validation_step():
    """Test that validation step works correctly."""
    print("\nTesting validation step...")
    
    model = DeepWeatherEvaluatorLightning(input_dim=5)
    
    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 5)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Test validation step
    loss = model.validation_step((x, y), batch_idx=0)
    
    print(f"  Validation loss: {loss.item():.4f}")
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    
    print("  [OK] Validation step test passed")


def test_optimizer_configuration():
    """Test that optimizer configuration works."""
    print("\nTesting optimizer configuration...")
    
    # Test different optimizers
    optimizers = ["adam", "adamw", "sgd"]
    
    for optimizer_name in optimizers:
        model = DeepWeatherEvaluatorLightning(
            input_dim=5,
            optimizer=optimizer_name,
            learning_rate=0.01,
        )
        
        optimizers_config = model.configure_optimizers()
        
        if isinstance(optimizers_config, dict):
            optimizer = optimizers_config["optimizer"]
        elif isinstance(optimizers_config, tuple):
            optimizer = optimizers_config[0][0]
        else:
            optimizer = optimizers_config
        
        print(f"  {optimizer_name}: {type(optimizer).__name__}")
        assert optimizer is not None, f"Optimizer for {optimizer_name} should not be None"
    
    print("  [OK] Optimizer configuration test passed")


def test_model_summary():
    """Test that model summary works."""
    print("\nTesting model summary...")
    
    model = DeepWeatherEvaluatorLightning(
        input_dim=5,
        hidden_dims=[40, 20, 10],
        dropout_rate=0.3,
        use_batch_norm=True,
    )
    
    summary = model.get_model_summary()
    
    print("  Model summary keys:", list(summary.keys()))
    
    expected_keys = [
        "input_dim", "output_dim", "hidden_dims", "total_params",
        "trainable_params", "dropout_rate", "use_batch_norm", "activation"
    ]
    
    for key in expected_keys:
        assert key in summary, f"Key {key} missing from summary"
    
    print(f"  Total parameters: {summary['total_params']:,}")
    print(f"  Trainable parameters: {summary['trainable_params']:,}")
    
    print("  [OK] Model summary test passed")


def quick_training_test():
    """Quick test of training with synthetic data."""
    print("\nQuick training test...")
    
    # Create synthetic data
    num_samples = 100
    num_features = 5
    
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, 2, (num_samples, 1)).float()
    
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = DeepWeatherEvaluatorLightning(
        input_dim=num_features,
        hidden_dims=[20, 10],
        learning_rate=0.01,
    )
    
    # Create a simple trainer for testing
    trainer = pl.Trainer(
        max_epochs=2,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    
    # Train for 2 epochs
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    
    print("  [OK] Quick training test passed")
    return trainer


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DeepWeatherEvaluatorLightning Implementation")
    print("=" * 60)
    
    try:
        # Run all tests
        model = test_model_creation()
        test_training_step()
        test_validation_step()
        test_optimizer_configuration()
        test_model_summary()
        trainer = quick_training_test()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully! [OK]")
        print("=" * 60)
        
        # Print final model info
        summary = model.get_model_summary()
        print(f"\nFinal model architecture:")
        print(f"  Input dimension: {summary['input_dim']}")
        print(f"  Hidden layers: {summary['hidden_dims']}")
        print(f"  Output dimension: {summary['output_dim']}")
        print(f"  Total parameters: {summary['total_params']:,}")
        print(f"  Using batch norm: {summary['use_batch_norm']}")
        print(f"  Dropout rate: {summary['dropout_rate']}")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)