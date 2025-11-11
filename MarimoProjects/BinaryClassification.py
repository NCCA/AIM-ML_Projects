import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import mlutils
    print(mlutils.__version__)
    return (mlutils,)


@app.cell
def _():
    from sklearn.datasets import make_circles
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return make_circles, nn, plt, torch


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Generation

    We will create a dataset of circles in a 2D array.
    """)
    return


@app.cell
def _(make_circles):
    n_samples = 1000
    x,y = make_circles(n_samples,noise=0.03,random_state=42)
    print(x.shape,y.shape)
    print(y[0:10])

    return x, y


@app.cell
def _(plt, x, y):
    plt.figure(figsize=(4,4))
    plt.scatter(x=x[:,0],y=x[:,1], c=y ,s=2, cmap=plt.cm.plasma)
    return


@app.cell
def _(torch, x, y):
    X = torch.from_numpy(x).type(torch.float)
    y_labels = torch.from_numpy(y).type(torch.float)
    train_split = int(0.8 * len(X))

    X_train = X[:train_split]
    X_test = X[train_split:]

    y_train = y[:train_split]
    y_test = y[train_split:]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


    return X_test, y_test


@app.cell
def _(mlutils):
    device = mlutils.get_device()
    print(device)
    return (device,)


@app.cell
def _(device, nn):
    class BinaryClassify(nn.Module) :
        def __init__(self) :
            super().__init__()
            # Layer 1 has 2 inputs (x,y for the dots) and we will expand this to 5 output (hidden layer)
            self.layer_1 = nn.Linear(in_features = 2 , out_features =5)
            # layer 2 must have the same input as previous layout output
            # We will have 1 output feature as it's either a 0 or 1
            self.layer_2 = nn.Linear(in_features = 5 , out_features=1)

        def forward(self,x) :
            return self.layer_2(self.layer_1(x))

    model = BinaryClassify().to(device)
    print(model.parameters)
    print(model.state_dict())
        
    return (model,)


@app.cell
def _(X_test, device, model, y_test):
    _pred = model(X_test.to(device))
    print(f"Length of predictions {len(_pred)} Shape {_pred.shape}")
    print(f"Length of test samples {len(y_test)} Shape {y_test.shape}")
    print(_pred[:10])
    print(y_test[:10])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
