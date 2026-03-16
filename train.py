import wandb
from torch.profiler import profile, ProfilerActivity
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
def main():
    config = load_config("experiment_3.yaml")
    seed_everything(config.trainer.seed)

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # mode="disabled",
        config=config.model_dump(),
    )

    train(config)


if __name__ == "__main__":
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    # ) as prof:
    main()
