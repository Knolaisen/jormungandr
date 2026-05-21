import argparse
import wandb
from codecarbon import track_emissions
from jormungandr.config.configuration import (
    load_config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from jormungandr.training.trainer import train
from jormungandr.utils.seed import seed_everything


@track_emissions(
    country_iso_code="NOR",
    project_name="fafnir_training",
    log_level="ERROR",
)
def main(config_file: str):
    config = load_config(config_file)
    seed_everything(config.trainer.seed)

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # mode="disabled",
        config=config.model_dump(),
        settings=wandb.Settings(code_dir="./src"),
        id=config.trainer.run_id if config.trainer.run_id else None,
        resume="must" if config.trainer.run_id else None,
    )

    train(config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    experiment = "experiment_2.yaml"
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help=f"Config file to load (e.g. {experiment})",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help=f"Config file to load (e.g. {experiment})",
    )
    args = parser.parse_args()
    main(args.config_flag or args.config or experiment)
