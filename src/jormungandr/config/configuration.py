from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import yaml

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


class TrainerConfig(BaseModel):
    epochs: int = Field(default=5, description="Number of training epochs")
    batch_size: int = Field(
        default=16, description="Batch size for training and validation"
    )
    seed: int = Field(default=42, description="Random seed for reproducible runs")
    encoder_learning_rate: float = Field(
        default=0.001, description="Learning rate for the encoder optimizer"
    )
    decoder_learning_rate: float = Field(
        default=0.001, description="Learning rate for the decoder optimizer"
    )
    output_head_learning_rate: float = Field(
        default=0.001, description="Learning rate for the output head optimizer"
    )
    optimizer: str = Field(
        default="Adam", description="Optimizer to use (e.g., 'Adam', 'SGD')"
    )
    log_interval: int = Field(
        default=10, description="Interval (in batches) for logging training progress"
    )
    loss: LossConfig = Field(
        default_factory=LossConfig, description="Configuration for the loss function"
    )

class DecoderConfig(BaseModel):
    freeze_decoder: bool = Field(
        default=False, description="Whether to freeze the decoder during training"
    )
    epoch_to_unfreeze_decoder: int = Field(
        default=0,
        description="Epoch number at which to unfreeze the decoder if it is initially frozen",
    )
    num_queries: int = Field(
        default=None, description="Number of object queries for the decoder"
    )
    hidden_dim: int = Field(
        default=256, description="Hidden dimension size for the decoder's query position embeddings"
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
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig,
        description="Configuration for the decoder used in Fafnir",
    )


class Config(BaseModel):
    trainer: TrainerConfig
    fafnir: FafnirConfig


def load_config(filename: str) -> Config:
    path = os.path.join(CONFIG_PATH, filename)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Config(**raw_config)
