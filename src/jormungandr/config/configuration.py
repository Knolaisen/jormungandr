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
    name: str = Field(default="GIoULoss", description="Name of the loss function to use (e.g., 'GIoULoss', 'CIoULoss')")
    class_cost: float = Field(default=1.0, description="Relative weight of the classification error in the matching cost")
    bbox_cost: float = Field(default=5.0, description="Relative weight of the L1 error of the bounding box coordinates in the matching cost")
    giou_cost: float = Field(default=2.0, description="Relative weight of the giou loss of the bounding box in the matching cost")


class TrainerConfig(BaseModel):
    epochs: int = Field(default=5, description="Number of training epochs")
    batch_size: int = Field(default=16, description="Batch size for training and validation")
    learning_rate: float = Field(default=0.001, description="Learning rate for the optimizer")
    optimizer: str = Field(default="Adam", description="Optimizer to use (e.g., 'Adam', 'SGD')")
    log_interval: int = Field(default=10, description="Interval (in batches) for logging training progress")
    loss: LossConfig = Field(default_factory=LossConfig, description="Configuration for the loss function")

class EncoderConfig(BaseModel):
    type: str = Field(default="Mamba", description="Type of encoder to use (e.g., 'Mamba', 'Transformer')")
    num_layers: int = Field(default=6, description="Number of layers in the encoder")

class FafnirConfig(BaseModel):
    input_size: int = Field(default=512, description="Input size for the Fafnir model")
    num_classes: int = Field(default=80, description="Number of classes for object detection")
    encoder: EncoderConfig = Field(default_factory=EncoderConfig, description="Configuration for the encoder used in Fafnir")

class Config(BaseModel):
    trainer: TrainerConfig
    fafnir: FafnirConfig



def load_config(filename: str) -> Config:
    path = os.path.join(CONFIG_PATH, filename)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Config(**raw_config)