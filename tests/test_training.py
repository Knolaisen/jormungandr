from torch.utils.data import DataLoader
from datasets.load import load_dataset
import pytest
import torch
from torch.optim import AdamW


from jormungandr.config.configuration import load_config
from jormungandr.dataset import _collate_fn
from jormungandr.fafnir import Fafnir
from jormungandr.training.trainer import run_validation, train_one_epoch
from jormungandr.training.criterion import build_criterion


def create_datasets(
    dataset_name: str = "detection-datasets/coco",
    cache_dir: str = "../data/",
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    ds = load_dataset(dataset_name, cache_dir=cache_dir)

    torch_train_ds = ds["train"].with_format("torch")
    torch_val_ds = ds["val"].with_format("torch")

    return torch_train_ds, torch_val_ds


@pytest.fixture(scope="module")
def model():
    return Fafnir()


@pytest.mark.slow
def test_run_validation(model):
    _, val_dataset = create_datasets()
    # Limit size of validation dataset for testing purposes
    val_dataset = val_dataset.select(range(10))
    batch_size = 2
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("config.yaml")
    criterion = build_criterion(config.trainer.loss.name)

    # only use a few batches for testing
    average_val_loss, average_time = run_validation(
        model,
        val_loader,
        criterion,
        device,
        config,
    )

    assert isinstance(average_val_loss, float)
    assert average_val_loss >= 0.0

    # Check model is still the same after validation (i.e., no training should have occurred)
    for param in model.parameters():
        assert not param.grad, (
            "Model parameters should not have gradients after validation"
        )


@pytest.mark.slow
def test_run_train_one_epoch(model):
    train_dataset, _ = create_datasets()
    # Limit size of validation dataset for testing purposes
    train_dataset = train_dataset.select(range(10))
    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("config.yaml")
    criterion = build_criterion(config.trainer.loss.name)
    optimizer = AdamW(model.parameters(), lr=config.trainer.learning_rate)

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
    for original_param, updated_param in zip(original_params, model.parameters()):
        assert not torch.equal(original_param, updated_param), (
            "Model parameters should have been updated during training"
        )
