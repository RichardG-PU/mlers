"""
Utility helpers: seeding, class weights, early stopping.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    weight_c = N_total / (N_classes * N_c)
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    n_classes = len(classes)
    weights = n_total / (n_classes * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
