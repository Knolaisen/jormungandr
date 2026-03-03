from dotenv import load_dotenv
import os

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


if not all([WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY]):
    raise ValueError(
        "Please set the WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY environment variables."
    )
