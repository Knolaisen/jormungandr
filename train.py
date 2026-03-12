import wandb

from jormungandr.config.configuration import (
    load_config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from jormungandr.training.trainer import train
from jormungandr.utils.seed import seed_everything


if __name__ == "__main__":
    config = load_config("config.yaml")
    seed_everything(config.trainer.seed)

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # mode="disabled",
        config=config.model_dump(),
    )

    train(config)
