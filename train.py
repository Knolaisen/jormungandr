import wandb

from jormungandr.config.configuration import (
    load_config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from jormungandr.training.trainer import train


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    config = load_config("config.yaml")

    train(config)
