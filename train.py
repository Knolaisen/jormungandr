import wandb

from jormungandr.config.configuration import (
    load_config,
    Config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from jormungandr.training.trainer import train


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    config = load_config("config.yaml")

    train(
        model=config.model,
        training_loader=config.data.training_loader,
        validation_loader=config.data.validation_loader,
        criterion=config.trainer.loss,
        optimizer=config.trainer.optimizer,
    )
