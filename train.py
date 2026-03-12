from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

from jormungandr.config.configuration import (
    load_config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from jormungandr.training.trainer import train


def main() -> None:
    config = load_config("config.yaml")
    accelerator = Accelerator(
        mixed_precision=config.trainer.accelerate.mixed_precision,
        log_with="wandb",
    )
    set_seed(config.trainer.seed)

    if accelerator.is_main_process:
        wandb.login(key=WANDB_API_KEY)

    accelerator.init_trackers(
        project_name=WANDB_PROJECT,
        config=config.model_dump(),
        init_kwargs={"wandb": {"entity": WANDB_ENTITY}},
    )

    train(config, accelerator=accelerator)
    accelerator.end_training()


if __name__ == "__main__":
    main()
