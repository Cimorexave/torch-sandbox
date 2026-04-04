import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Optional, List, Dict, Any
import numpy as np


class DeepWeatherEvaluatorLightning(pl.LightningModule):
    """
    PyTorch Lightning version of DeepWeatherEvaluator with improved architecture
    and training features.
    
    Features:
    - Configurable hidden layers
    - Dropout for regularization
    - Batch normalization option
    - Multiple activation functions
    - Automatic logging and metrics
    - Checkpointing and early stopping support
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 1,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = False,
        activation: str = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        scheduler: str = "reduce_lr",
        threshold: float = 0.5,
    ):
        """
        Initialize the Lightning model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output dimensions (1 for binary classification)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'tanh', 'sigmoid')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization)
            optimizer: Optimizer name ('adam', 'sgd', 'adamw')
            scheduler: Learning rate scheduler ('reduce_lr', 'cosine', 'step')
            threshold: Threshold for binary classification
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [input_dim * 8, input_dim * 4]
        
        self.hidden_dims = hidden_dims
        self.threshold = threshold
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Add dropout (except after last layer)
            if i < len(hidden_dims) - 1 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # For binary classification
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Setup metrics
        self._setup_metrics()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _setup_metrics(self):
        """Setup metrics for training, validation, and testing."""
        # Training metrics
        self.train_accuracy = Accuracy(task="binary", threshold=self.threshold)
        self.train_precision = Precision(task="binary", threshold=self.threshold)
        self.train_recall = Recall(task="binary", threshold=self.threshold)
        self.train_f1 = F1Score(task="binary", threshold=self.threshold)
        
        # Validation metrics
        self.val_accuracy = Accuracy(task="binary", threshold=self.threshold)
        self.val_precision = Precision(task="binary", threshold=self.threshold)
        self.val_recall = Recall(task="binary", threshold=self.threshold)
        self.val_f1 = F1Score(task="binary", threshold=self.threshold)
        
        # Test metrics
        self.test_accuracy = Accuracy(task="binary", threshold=self.threshold)
        self.test_precision = Precision(task="binary", threshold=self.threshold)
        self.test_recall = Recall(task="binary", threshold=self.threshold)
        self.test_f1 = F1Score(task="binary", threshold=self.threshold)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def predict_proba(self, x):
        """Predict probabilities."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x):
        """Predict binary classes."""
        proba = self.predict_proba(x)
        return (proba > self.threshold).float()
    
    def training_step(self, batch, batch_idx):
        """Training step with loss and metrics calculation."""
        x, y = batch
        y_hat = self.forward(x)
        
        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Update metrics
        self.train_accuracy(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=False)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        self.log("train_acc_epoch", self.train_accuracy.compute(), prog_bar=False)
        self.log("train_precision_epoch", self.train_precision.compute(), prog_bar=False)
        self.log("train_recall_epoch", self.train_recall.compute(), prog_bar=False)
        self.log("train_f1_epoch", self.train_f1.compute(), prog_bar=False)
        
        # Reset metrics for next epoch
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self.forward(x)
        
        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Update metrics
        self.val_accuracy(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        self.log("val_acc_epoch", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_precision_epoch", self.val_precision.compute(), prog_bar=False)
        self.log("val_recall_epoch", self.val_recall.compute(), prog_bar=False)
        self.log("val_f1_epoch", self.val_f1.compute(), prog_bar=False)
        
        # Reset metrics for next epoch
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self.forward(x)
        
        # Calculate loss
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Update metrics
        self.test_accuracy(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)
        
        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", self.test_accuracy)
        
        return loss
    
    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        test_acc = self.test_accuracy.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()
        
        self.log("test_acc_epoch", test_acc)
        self.log("test_precision_epoch", test_precision)
        self.log("test_recall_epoch", test_recall)
        self.log("test_f1_epoch", test_f1)
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        
        # Reset metrics
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Select optimizer
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Select scheduler
        if self.hparams.scheduler == "reduce_lr":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,
                eta_min=1e-6
            )
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def get_model_summary(self):
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            "input_dim": self.hparams.input_dim,
            "output_dim": self.hparams.output_dim,
            "hidden_dims": self.hidden_dims,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dropout_rate": self.hparams.dropout_rate,
            "use_batch_norm": self.hparams.use_batch_norm,
            "activation": self.hparams.activation,
        }
        
        return summary