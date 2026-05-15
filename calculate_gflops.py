import argparse
from codecarbon import track_emissions
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn
import torch
import wandb

from jormungandr.config.configuration import (
    load_config,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from jormungandr.fafnir import Fafnir
from jormungandr.jormungandr import Jormungandr
from jormungandr.config.configuration import FafnirConfig, JormungandrConfig
from jormungandr.utils.seed import seed_everything


def calculate_gflops(model: nn.Module, input_size: tuple) -> None:
    model.eval()
    dummy = torch.randn(*input_size).cuda()
    flops = FlopCountAnalysis(model, dummy)
    print(flop_count_table(flops, max_depth=3))
    print(f"Total GFLOPs: {flops.total() / 1e9:.2f}")
    print("Unsupported ops:", flops.unsupported_ops())
    wandb.log({"GFLOPs": flops.total() / 1e9})
    wandb.log({"Unsupported ops": flops.unsupported_ops()})


@track_emissions(
    country_iso_code="NOR",
    project_name="fafnir_training",
    log_level="ERROR",
)
def main(config_file: str) -> None:
    config = load_config(config_file)
    seed_everything(config.trainer.seed)

    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # mode="disabled",
        config=config.model_dump(),
    )
    device = "cuda"

    if isinstance(config.model, JormungandrConfig):
        model = Jormungandr(config=config.model).to(device)
    elif isinstance(config.model, FafnirConfig):
        model = Fafnir(config=config.model).to(device)

    calculate_gflops(model, input_size=(1, 3, 800, 1333))
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
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=None,
        help="Path to the trained model checkpoint to validate",
    )
    args = parser.parse_args()
    main(args.config_flag or args.config or experiment)
