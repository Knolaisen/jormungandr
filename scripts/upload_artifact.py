import wandb
from jormungandr.config.configuration import (
    load_config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)

wandb.login(key=WANDB_API_KEY)
wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # mode="disabled",
        settings=wandb.Settings(code_dir="./src"),
    )



artifact = wandb.Artifact(
    name="mamba2-giou",   # same name as before if you want
    type="model"          # or "model", "results", etc.
)

artifact.add_file("models/mamba_giouloss_20260423_172048")

wandb.log_artifact(artifact)
wandb.finish()