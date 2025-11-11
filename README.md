## MLProjects

The idea of this setup is to provide a full working environment for using Marimo and our own library of useful functions. This can be replicated easily in any folder and is idea for when you need to start work on your assignments.

This folder is going to contain two directories, one with a "Utils library" for re-usable functions and one with all our marimo notebooks in.

This setup can be easily re-used for all our learning projects and similar for our assignments.

## Step 1 create the ML Folder

This folder must be in a new non parented .venv folder (i.e. Desktop/) as it will contain other  .venvs and projects. First we will create the base folder

```zsh
mkdir MLProjects
cd MLProjects
```

All the work we now do will be from within this base folder.

## Step 2 MLUtils Folder

From the MLProjects folder we will create a new python package called MLUtils and add functions we need in it.

```zsh
uv init --package MLUtils
cd MLUtils
touch src/Utils.py
```

This will create the following structure

```
tree
.
├── pyproject.toml
├── README.md
└── src
    └── mlutils
        ├── __init__.py
        └── Utils.py
```


We can edit the __init__.py file to add the following


```python
__version__ = "0.0.5"

from .Utils import get_device

__all__ = [get_device]
```

And the Utils.py file to add the following


```python
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
```

Over the next few weeks of projects we will add more to this file.

## Step 3 MarimoProjects

We will create a new folder to use for our Marimo projects. This will use the MarimoDir.sh script and setup most of the basics we need.

In the root project folder (MLProjects) type the following

```zsh
MarimoDir.sh MarimoProjects
cd MarimoProjects
```

This will create the following files to start with

```
.
├── .envrc
├── pyproject.toml
├── uv.lock
├── .venv
└── zsh_functions.sh
```

We will use this for all our ML projects, but for now we will add some basic elements. Edit the pyproject.toml file and add the following

```toml
dependencies = [
    "marimo>=0.17.7",
    "ty>=0.0.1a26",
    "MLUtils"
]

[tool.uv.sources]
MLUtils = {path="../MLUtils", editable = true}
```

We can now run ```uv sync``` and it should pick up the MLUtils module we have developed.

## Other Packages

So far in the labs we have added the following packages, we will add more over the next few weeks.

```zsh
uv add torch numpy matplotlib scikit-learn ruff
```

The rest of the work we do will be added to this repository.
