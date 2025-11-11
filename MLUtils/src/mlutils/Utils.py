import torch


def get_device() -> torch.device:
    """
    Returns the appropriate device for our current environment. If GPU can be found it will
    use this.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
