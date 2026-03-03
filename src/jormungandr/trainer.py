"""
The trainer script is responsible for training the Jormungandr model. It includes functions for loading data, defining the training loop, and evaluating the model's performance. The trainer will utilize the output head defined in `output_head.py` to compute the loss and update the model's parameters during training. It will also handle saving and loading model checkpoints, as well as logging training metrics for analysis.
It shall be able to train the model on a given dataset, and evaluate its performance on a validation set. The trainer will also include functionality for hyperparameter tuning and early stopping based on validation performance. Additionally, it will support distributed training across multiple GPUs if available.
It shall be able to customize the training process with different optimizers, learning rate schedulers, and loss functions. The trainer will also include functionality for visualizing training progress and results, such as plotting loss curves and displaying sample predictions on the validation set. Overall, the trainer will be a crucial component in the development and optimization of the Jormungandr model for object detection tasks.
It shall be able to train both the Fafnir and Jormungandr models

It shall use:
* Torch
* Torchvision for data loading and augmentation
* Wandb for logging and visualization
*
"""

from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/trainer_{timestamp}")
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.0


def train(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module | Callable,
    optimizer: optim.Optimizer,
):
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1} starting...")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            model, training_loader, optimizer, criterion, device="cuda"
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                validation_image, validation_labels = vdata
                voutputs = model(validation_image)
                val_loss = criterion(voutputs, validation_labels)
                running_vloss += val_loss

        average_validation_loss = running_vloss / (i + 1)
        print(f"LOSS train {avg_loss:.3f} valid {average_validation_loss:.3f}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": average_validation_loss},
            epoch + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            model_path = "model_{}_{}".format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    return model


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f"Batch {i + 1} loss: {last_loss:.3f}")
            running_loss = 0.0

    return last_loss
