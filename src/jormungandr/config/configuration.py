import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

CONFIG_PATH = os.path.dirname(__file__)

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


if not all([WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY]):
    raise ValueError(
        "Please set the WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY environment variables."
    )


class LossConfig(BaseModel):
    name: str = Field(
        default="GIoULoss",
        description="Name of the loss function to use (e.g., 'GIoULoss', 'CIoULoss')",
    )
    class_cost: float = Field(
        default=1.0,
        description="Relative weight of the classification error in the matching cost",
    )
    bbox_cost: float = Field(
        default=5.0,
        description="Relative weight of the L1 error of the bounding box coordinates in the matching cost",
    )
    giou_cost: float = Field(
        default=2.0,
        description="Relative weight of the giou loss of the bounding box in the matching cost",
    )

    num_labels: int = Field(
        default=91, description="Number of class labels (including background)"
    )
    eos_coefficient: float = Field(
        default=0.1,
        description="Relative weight of the no-object class in the classification cost",
    )
    auxiliary_loss: bool = Field(
        default=False,
        description="Whether to compute auxiliary losses from intermediate layers of the model",
    )
    bbox_loss_coefficient: float = Field(
        default=5.0,
        description="Coefficient for the bounding box L1 loss in the total loss calculation",
    )
    giou_loss_coefficient: float = Field(
        default=2.0,
        description="Coefficient for the GIoU loss in the total loss calculation",
    )


class AccelerateConfig(BaseModel):
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(
        default="no", description="Mixed precision mode for distributed training"
    )


class TrainerConfig(BaseModel):
    epochs: int = Field(default=5, description="Number of training epochs")
    batch_size: int = Field(
        default=16, description="Batch size for training and validation"
    )
    seed: int = Field(default=42, description="Random seed for reproducible runs")
    learning_rate: float = Field(
        default=0.001, description="Learning rate for the optimizer"
    )
    optimizer: str = Field(
        default="Adam", description="Optimizer to use (e.g., 'Adam', 'SGD')"
    )
    log_interval: int = Field(
        default=10, description="Interval (in batches) for logging training progress"
    )
    num_workers: int = Field(
        default=4, ge=0, description="Number of workers per dataloader process"
    )
    pin_memory: bool = Field(
        default=True, description="Whether dataloaders should pin host memory"
    )
    persistent_workers: bool = Field(
        default=True, description="Keep dataloader workers alive between epochs"
    )
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description="How many batches each worker should prefetch ahead of time",
    )
    gradient_clip_norm: float = Field(
        default=1.0, gt=0, description="Gradient clipping threshold"
    )
    loss: LossConfig = Field(
        default_factory=LossConfig, description="Configuration for the loss function"
    )
    accelerate: AccelerateConfig = Field(
        default_factory=AccelerateConfig,
        description="Configuration for Hugging Face Accelerate",
    )


class EncoderConfig(BaseModel):
    type: str = Field(
        default="Mamba",
        description="Type of encoder to use (e.g., 'Mamba', 'Transformer')",
    )
    num_layers: int = Field(default=6, description="Number of layers in the encoder")


class FafnirConfig(BaseModel):
    input_size: int = Field(default=512, description="Input size for the Fafnir model")
    num_classes: int = Field(
        default=80, description="Number of classes for object detection"
    )
    encoder: EncoderConfig = Field(
        default_factory=EncoderConfig,
        description="Configuration for the encoder used in Fafnir",
    )


class Config(BaseModel):
    trainer: TrainerConfig
    fafnir: FafnirConfig


def load_config(filename: str) -> Config:
    path = os.path.join(CONFIG_PATH, filename)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Config(**raw_config)
