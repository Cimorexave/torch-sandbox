# DeepWeatherEvaluator with PyTorch Lightning

An improved version of the DeepWeatherEvaluator model implemented using PyTorch Lightning. The new implementation provides better organization, more features, and easier training compared to the original `dwe.py`.

## Files

1. **`dwe_lightning.py`** - The main PyTorch Lightning model
2. **`example_usage_lightning.py`** - Comprehensive example with training, validation, and testing
3. **`test_lightning.py`** - Unit tests for the Lightning implementation
4. **`dwe.py`** - Original implementation (for comparison)
5. **`example_usage.py`** - Original example usage (for comparison)


## Quick Start

### Installation
```bash
pip install pytorch-lightning torchmetrics
```
Or
``` poetry add pytorch pytorch-lightning torchmetrics
```

### Basic Usage

```python
from dwe_lightning import DeepWeatherEvaluatorLightning
import pytorch_lightning as pl

# Create model
model = DeepWeatherEvaluatorLightning(
    input_dim=5,
    hidden_dims=[40, 20, 10],
    dropout_rate=0.3,
    use_batch_norm=True,
    learning_rate=0.001,
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="auto",
    devices="auto",
)

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Test the model
trainer.test(model, dataloaders=test_loader)
```

### Using the DataModule

```python
from example_usage_lightning import WeatherDataModule

# Create data module
data_module = WeatherDataModule(
    num_samples=1000,
    num_features=5,
    batch_size=32,
    val_split=0.2,
    test_split=0.2,
)

# Prepare and setup data
data_module.prepare_data()
data_module.setup()

# Train with data module
trainer.fit(model, datamodule=data_module)
```

## Advanced Features

### Hyperparameter Tuning
```python
configs = [
    {"hidden_dims": [40, 20], "dropout_rate": 0.2, "activation": "relu"},
    {"hidden_dims": [64, 32, 16], "dropout_rate": 0.3, "activation": "leaky_relu"},
    {"hidden_dims": [32, 16, 8], "dropout_rate": 0.5, "activation": "tanh"},
]

for config in configs:
    model = DeepWeatherEvaluatorLightning(**config)
    # Train and evaluate...
```

### Callbacks and Logging
```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

callbacks = [
    ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=3),
    EarlyStopping(monitor="val_loss", patience=10),
    LearningRateMonitor(),
]

loggers = [
    CSVLogger("logs"),
    TensorBoardLogger("logs"),
]

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    logger=loggers,
)
```

### Model Inspection
```python
# Get model summary
summary = model.get_model_summary()
print(f"Total parameters: {summary['total_params']:,}")
print(f"Hidden layers: {summary['hidden_dims']}")

# Make predictions
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
```

## Running Examples

1. **Quick test**: `poetry run test_lightning`
2. **Full example**: `poetry run example_usage_lightning`
3. **Original example**: `poetry run example_usage.py` (for comparison)