"""
Pydantic configuration dataclasses for Fafnir and Jormungandr.

Loads W&B credentials from environment variables at import time and raises
immediately if they are missing. YAML configs are resolved relative to this
file's directory via `load_config`.

Classes:
    LossConfig         -- loss weights and auxiliary-loss settings.
    SchedulerConfig    -- LR scheduler name and extra kwargs.
    TrainerConfig      -- full training-loop settings (LR, batch size, etc.).
    BackboneConfig     -- backbone freeze flag and model-name.
    EncoderConfig      -- encoder type, layer count, and Mamba-specific dims.
    DecoderConfig      -- decoder freeze, query count, and pre-trained flag.
    OutputHeadConfig   -- output-head freeze and pre-trained flag.
    FafnirConfig       -- single-frame model config (extends DETRConfig).
    JormungandrConfig  -- video model config with spatial/temporal encoders.
    Config             -- top-level config tying trainer and model together.

Functions:
    load_config -- parse a YAML file from this directory into a Config.
"""

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
    decoder_layers: int = Field(
        default=6,
        description="Number of decoder layers in the model, used to determine how many auxiliary losses to compute if auxiliary_loss is True",
    )


class SchedulerConfig(BaseModel):
    name: str = Field(
        default="CosineAnnealingLR",
        description="Name of the LR scheduler (e.g., 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'OneCycleLR')",
    )
    params: dict = Field(
        default_factory=dict,
        description="Extra keyword arguments passed directly to the scheduler constructor (e.g., {T_max: 50, eta_min: 1e-6})",
    )


class TrainerConfig(BaseModel):
    epochs: int = Field(default=5, description="Number of training epochs")
    batch_size: int = Field(
        default=16, description="Batch size for training and validation"
    )
    val_batch_size: int = Field(
        default=1,
        description="Batch size for validation (can be different from training batch size if desired)",
    )
    seed: int = Field(default=42, description="Random seed for reproducible runs")
    subset_size: int | None = Field(
        default=None,
        description="If set, limits the number of samples used for training and validation to this number (useful for quick testing)",
    )
    backbone_learning_rate: float = Field(
        default=5e-6, description="Learning rate for the backbone optimizer"
    )
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
    num_log_images: int = Field(
        default=8,
        description="Number of validation images to log with bounding boxes per epoch",
    )
    viz_score_threshold: float = Field(
        default=0.5,
        description="Minimum confidence score for predicted boxes to be visualized",
    )
    loss: LossConfig = Field(
        default_factory=LossConfig, description="Configuration for the loss function"
    )
    scheduler: SchedulerConfig | None = Field(
        default=None,
        description="Configuration for the learning rate scheduler. If None, no scheduler is used.",
    )
    epoch_to_unfreeze_decoder: int = Field(
        default=0,
        description="Epoch number at which to unfreeze the decoder if it is initially frozen",
    )
    epoch_to_unfreeze_backbone: int = Field(
        default=500,
        description="Epoch number at which to unfreeze the backbone if it is initially frozen",
    )
    epoch_to_unfreeze_output_head: int = Field(
        default=0,
        description="Epoch number at which to unfreeze the output head if it is initially frozen",
    )

    # Dataset and dataloader parameters
    dataset_name: str = Field(
        default="coco",
        description="Name of the dataset to use (e.g., 'coco', 'mot17')",
    )


class DecoderConfig(BaseModel):
    freeze_decoder: bool = Field(
        default=False, description="Whether to freeze the decoder during training"
    )
    num_queries: int | None = Field(
        default=None, description="Number of object queries for the decoder"
    )
    hidden_dim: int = Field(
        default=256,
        description="Hidden dimension size for the decoder's query position embeddings",
    )
    auxiliary_loss: bool = Field(
        default=True,
        description="Whether the decoder is configured to compute auxiliary losses from intermediate layers",
    )
    use_pre_trained: bool = Field(
        default=True,
        description="Whether to use a pre-trained decoder (e.g., from a DETR model) or a custom Mamba decoder",
    )


class EncoderConfig(BaseModel):
    encoder_type: str = Field(
        default="Mamba",
        description="Type of encoder to use (e.g., 'Mamba', 'Transformer')",
    )
    num_layers: int = Field(default=6, description="Number of layers in the encoder")
    use_pre_trained: bool = Field(
        default=True,
        description="Whether to use a pre-trained encoder (e.g., from a DETR model) or a custom Mamba encoder",
    )
    dim_feedforward: int = Field(
        default=2048,
        description="Dimension of the feedforward network in the MambaEncoderFFN variant (ignored if encoder_type is not 'MambaFFN')",
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout rate for the feedforward network in the MambaEncoderFFN variant (ignored if encoder_type is not 'MambaFFN')",
    )
    hidden_state_dim: int = Field(
        default=64,
        description="SSM state dimension (d_state). Mamba-2 recommended range: 64–256. Mamba-1 default of 16 under-utilizes Mamba-2.",
    )
    model_dimension: int = Field(
        default=256,
        description="Model/token embedding dimension (d_model).",
    )
    mamba_variant: str = Field(
        default="mamba2",
        description="Variant of Mamba to use if encoder_type is 'Mamba' or 'MambaFFN' (e.g., 'mamba1', 'mamba2')",
    )


class OutputHeadConfig(BaseModel):
    freeze_prediction_head: bool = Field(
        default=False, description="Whether to freeze the output head during training"
    )
    use_pre_trained: bool = Field(
        default=True,
        description="Whether to use a pre-trained output head (e.g., from a DETR model) or a custom FCNN prediction head",
    )


class BackboneConfig(BaseModel):
    model_name: str = Field(
        default="facebook/detr-resnet-50",
        description="Name of the pre-trained DETR model to use for the backbone (e.g., 'facebook/detr-resnet-50')",
    )
    freeze_backbone: bool = Field(
        default=True, description="Whether to freeze the backbone during training"
    )


class DETRConfig(BaseModel):
    input_size: int = Field(default=512, description="Input size for the Fafnir model")
    num_classes: int = Field(
        default=80, description="Number of classes for object detection"
    )
    model_dimension: int = Field(
        default=256, description="Model/token embedding dimension (d_model)"
    )
    backbone: BackboneConfig = Field(
        default_factory=BackboneConfig,
        description="Configuration for the backbone used in Fafnir",
    )
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig,
        description="Configuration for the decoder used in Fafnir",
    )
    output_head: OutputHeadConfig = Field(
        default_factory=OutputHeadConfig,
        description="Configuration for the output head used in Fafnir",
    )
    detr_name: str = Field(
        default="facebook/detr-resnet-50",
        description="Name of the pre-trained DETR model to use for the encoder and decoder (e.g., 'facebook/detr-resnet-50', 'facebook/detr-resnet-101', 'facebook/detr-resnet-50-dc5' or 'facebook/detr-resnet-101-dc5')",
    )
    checkpoint_name: str | None = Field(
        default=None,
        description="Name of a checkpoint artifact stored in Weight & Biases to load model weights from (e.g., 'jormungandr/fafnir-checkpoint:v0'). If None, no checkpoint is loaded and the model is trained from scratch.",
    )


class FafnirConfig(DETRConfig):
    encoder: EncoderConfig = Field(
        default_factory=EncoderConfig,
        description="Configuration for the encoder used in Fafnir",
    )


class JormungandrConfig(DETRConfig):
    num_frames: int = Field(
        default=4,
        description="Number of frames in the input video clip for Jormungandr",
    )
    spatial_encoder: EncoderConfig = Field(
        default_factory=EncoderConfig,
        description="Configuration for the spatial encoder used in Jormungandr (operates on individual frames)",
    )
    temporal_encoder: EncoderConfig = Field(
        default_factory=EncoderConfig,
        description="Configuration for the temporal encoder used in Jormungandr (operates across frames)",
    )


class Config(BaseModel):
    trainer: TrainerConfig
    model: FafnirConfig | JormungandrConfig


def load_config(filename: str) -> Config:
    path = os.path.join(CONFIG_PATH, filename)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Config(**raw_config)
