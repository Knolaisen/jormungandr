from torch.optim import AdamW, Optimizer


def build_optimizer(name: str) -> Optimizer:
    if name == "AdamW":
        return AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
