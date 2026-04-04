import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dwe_lightning import DeepWeatherEvaluatorLightning


class WeatherDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for weather data.
    Handles data preparation, splitting, and dataloader creation.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_features: int = 5,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.2,
        random_seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Data attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Generate synthetic data."""
        np.random.seed(self.random_seed)
        
        labels = np.random.randint(0, 2, size=(self.num_samples, 1))
        features = np.random.randn(self.num_samples, self.num_features)
        
        # Add some signal: make first feature correlate with label
        features[:, 0] += labels.squeeze() * 0.5
        
        # Combine features and labels
        self.data = np.hstack([features, labels])
        
        print(f"Generated synthetic data with shape: {self.data.shape}")
        print(f"Class distribution: {np.bincount(labels.squeeze().astype(int))}")
        
    def setup(self, stage: str = None):
        """Split data into train, validation, and test sets."""
        # Split features and labels
        X = self.data[:, :-1].astype(np.float32)
        y = self.data[:, -1:].astype(np.float32)  # Keep as 2D array
        
        # Calculate split sizes
        test_size = int(self.num_samples * self.test_split)
        val_size = int(self.num_samples * self.val_split)
        train_size = self.num_samples - test_size - val_size
        
        # Split indices
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = TensorDataset(
            torch.from_numpy(X[train_idx]),
            torch.from_numpy(y[train_idx])
        )
        
        self.val_dataset = TensorDataset(
            torch.from_numpy(X[val_idx]),
            torch.from_numpy(y[val_idx])
        )
        
        self.test_dataset = TensorDataset(
            torch.from_numpy(X[test_idx]),
            torch.from_numpy(y[test_idx])
        )
        
        print(f"\nData splits:")
        print(f"  Training: {len(self.train_dataset)} samples")
        print(f"  Validation: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )


def train_model():
    """Train the model with PyTorch Lightning."""
    print("=" * 60)
    print("Training DeepWeatherEvaluator with PyTorch Lightning")
    print("=" * 60)
    
    # Initialize data module
    data_module = WeatherDataModule(
        num_samples=2000,
        num_features=5,
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup()
    
    # Initialize model with hyperparameters
    model = DeepWeatherEvaluatorLightning(
        input_dim=5,
        output_dim=1,
        hidden_dims=[40, 20, 10],  # More flexible architecture
        dropout_rate=0.3,
        use_batch_norm=True,
        activation="leaky_relu",
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="reduce_lr",
        threshold=0.5,
    )
    
    # Print model summary
    summary = model.get_model_summary()
    print("\nModel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc_epoch",
        mode="max",
        filename="dwe-{epoch:02d}-{val_acc_epoch:.3f}",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Setup loggers
    csv_logger = CSVLogger("logs", name="dwe_lightning")
    tensorboard_logger = TensorBoardLogger("logs", name="dwe_lightning")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.fit(model, datamodule=data_module)
    
    # Test the model
    print("\nTesting model...")
    trainer.test(model, datamodule=data_module)
    
    # Load best model and test again
    if checkpoint_callback.best_model_path:
        print(f"\nLoading best model from: {checkpoint_callback.best_model_path}")
        best_model = DeepWeatherEvaluatorLightning.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        trainer.test(best_model, datamodule=data_module)
    
    return model, trainer


def hyperparameter_tuning_example():
    """Example of hyperparameter tuning with different configurations."""
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Example")
    print("=" * 60)
    
    # Different configurations to try
    configs = [
        {
            "name": "Baseline",
            "hidden_dims": [40, 20],
            "dropout_rate": 0.2,
            "use_batch_norm": False,
            "activation": "relu",
            "learning_rate": 0.001,
        },
        {
            "name": "Deep Network",
            "hidden_dims": [64, 32, 16, 8],
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "activation": "leaky_relu",
            "learning_rate": 0.0005,
        },
        {
            "name": "Regularized",
            "hidden_dims": [32, 16],
            "dropout_rate": 0.5,
            "use_batch_norm": True,
            "activation": "tanh",
            "learning_rate": 0.002,
        },
    ]
    
    # Initialize data module
    data_module = WeatherDataModule(
        num_samples=1000,
        num_features=5,
        batch_size=32,
        val_split=0.2,
        test_split=0.2,
    )
    data_module.prepare_data()
    data_module.setup()
    
    results = []
    
    for config in configs:
        print(f"\nTraining {config['name']} configuration...")
        
        model = DeepWeatherEvaluatorLightning(
            input_dim=5,
            output_dim=1,
            hidden_dims=config["hidden_dims"],
            dropout_rate=config["dropout_rate"],
            use_batch_norm=config["use_batch_norm"],
            activation=config["activation"],
            learning_rate=config["learning_rate"],
            optimizer="adam",
            scheduler="reduce_lr",
        )
        
        trainer = pl.Trainer(
            max_epochs=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            devices="auto",
        )
        
        # Train quickly
        trainer.fit(model, datamodule=data_module)
        
        # Test
        test_result = trainer.test(model, datamodule=data_module)
        test_acc = test_result[0]["test_acc_epoch"]
        
        results.append({
            "config": config["name"],
            "test_accuracy": test_acc,
            "params": sum(p.numel() for p in model.parameters()),
        })
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Parameters: {results[-1]['params']:,}")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Results")
    print("=" * 60)
    for result in results:
        print(f"{result['config']:15} | Accuracy: {result['test_accuracy']:.4f} | Params: {result['params']:,}")


def inference_example(model, data_module):
    """Example of using the trained model for inference."""
    print("\n" + "=" * 60)
    print("Inference Example")
    print("=" * 60)
    
    # Get some test data
    test_loader = data_module.test_dataloader()
    batch = next(iter(test_loader))
    x_batch, y_batch = batch
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        probabilities = model.predict_proba(x_batch[:10])  # First 10 samples
        predictions = model.predict(x_batch[:10])
    
    print("\nSample predictions:")
    print("  True labels | Predicted probabilities | Predicted classes")
    print("  " + "-" * 60)
    
    for i in range(min(10, len(x_batch))):
        true_label = y_batch[i].item()
        prob = probabilities[i].item()
        pred = predictions[i].item()
        
        print(f"  {true_label:11.0f} | {prob:23.4f} | {pred:18.0f}")
    
    # Calculate overall test accuracy
    test_loader = data_module.test_dataloader()
    all_preds = []
    all_labels = []
    
    for x, y in test_loader:
        preds = model.predict(x)
        all_preds.append(preds)
        all_labels.append(y)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"\nOverall test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    print("DeepWeatherEvaluator Lightning - Example Usage")
    print("=" * 60)
    
    # Option 1: Full training with logging and callbacks
    print("\n1. Full Training Example")
    trained_model, trainer = train_model()
    
    # Option 2: Hyperparameter tuning example
    print("\n2. Hyperparameter Tuning Example")
    hyperparameter_tuning_example()
    
    # Note: For inference example, we would need to recreate the data module
    # or save it from the training run. For simplicity, we'll create a new one.
    print("\n3. Inference Example")
    data_module = WeatherDataModule(
        num_samples=500,
        num_features=5,
        batch_size=32,
    )
    data_module.prepare_data()
    data_module.setup()
    
    # Create a simple model for inference demo
    demo_model = DeepWeatherEvaluatorLightning(
        input_dim=5,
        output_dim=1,
        hidden_dims=[40, 20],
        dropout_rate=0.2,
    )
    
    # Quick train for demo purposes
    demo_trainer = pl.Trainer(
        max_epochs=5,
        enable_progress_bar=False,
        logger=False,
    )
    demo_trainer.fit(demo_model, datamodule=data_module)
    
    inference_example(demo_model, data_module)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the 'logs' directory for training logs")
    print("2. Use TensorBoard: tensorboard --logdir logs/")
    print("3. Experiment with different hyperparameters")
    print("4. Try with real weather data")