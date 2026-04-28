from jormungandr.datasets import create_dataloaders
from torch.utils.data import DataLoader
from datasets.load import load_dataset
import pytest
import torch
from torch.optim import AdamW


from jormungandr.config.configuration import load_config

# from jormungandr.dataset import _collate_fn
from jormungandr.fafnir import Fafnir
from jormungandr.training.trainer import run_validation, train_one_epoch
from jormungandr.training.criterion import build_criterion


@pytest.fixture(scope="module")
def model():
    return Fafnir()


@pytest.mark.slow
def test_run_validation(model):
    subset_size = 10
    batch_size = 2
    train_loader, val_loader = create_dataloaders(
        subset_size=subset_size, batch_size=batch_size
    )
    # Limit size of validation dataset for testing purposes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("config.yaml")
    criterion = build_criterion(config.trainer.loss.name)

    # only use a few batches for testing
    average_val_loss, average_time, val_ap = run_validation(
        model,
        val_loader,
        criterion,
        device,
        config,
    )

    assert isinstance(average_val_loss, float)
    assert average_val_loss >= 0.0

    assert isinstance(average_time, float)
    assert average_time >= 0.0

    assert isinstance(val_ap, float)
    assert 0.0 <= val_ap <= 1.0

    # Check model is still the same after validation (i.e., no training should have occurred)
    for param in model.parameters():
        assert not param.grad, (
            "Model parameters should not have gradients after validation"
        )


@pytest.mark.slow
def test_run_train_one_epoch(model):
    subset_size = 10
    batch_size = 2
    train_loader, val_loader = create_dataloaders(
        subset_size=subset_size, batch_size=batch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("config.yaml")
    criterion = build_criterion(config.trainer.loss.name)
    optimizer = AdamW(model.parameters(), lr=config.trainer.encoder_learning_rate)

    # Copy model parameters before training to check for updates later
    original_params = [param.clone() for param in model.parameters()]
    # only use a few batches for testing
    average_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        config,
    )

    assert isinstance(average_loss, float), (
        f"Expected average_loss to be a float, got {type(average_loss)}"
    )
    assert average_loss >= 0.0

    # Check that model parameters have been updated (i.e., training should have occurred)
    params_changed = any(
        not torch.equal(orig, new)
        for orig, new in zip(original_params, model.parameters())
    )

    assert params_changed, (
        "At least one model parameter should have been updated during training"
    )
