import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pathlib
    import sys
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import Dataset,DataLoader
    import zipfile
    import string
    import pandas as pd
    import mlutils
    from ASLDataSet import ASLDataSet
    return (mlutils,)


@app.cell
def _(mlutils):
    device = mlutils.device()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
