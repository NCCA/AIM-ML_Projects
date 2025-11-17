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
    return make_circles, nn, optim, plt, torch


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

    y_train = y_labels[:train_split]
    y_test = y_labels[train_split:]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_test, X_train, y_test, y_train


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Function and Optimizer

    We are doing binary classification so there are built in optimizers and loss functions for this.

    BCELosss or BCELossWithLoigtsLoss
    """)
    return


@app.cell
def _(model, nn, optim):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params = model.parameters(), lr=0.1)
    return loss_fn, optimizer


@app.cell
def _(torch):
    def accuracy(y_true , y_pred) :
        correct = torch.eq(y_true,y_pred).sum()
        if len(y_pred) > 0 :
            acc = (correct.item() / len(y_pred)) * 100
            return acc
        else :
            return 0.0
    
    
    return (accuracy,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy,
    device,
    loss_fn,
    model,
    optimizer,
    torch,
    y_test,
    y_train,
):
    epochs = 100
    features_train_gpu = X_train.to(device)
    features_test_gpu = X_test.to(device)
    labels_train_gpu = y_train.to(device)
    labels_test_gpu = y_test.to(device)

    for epoch in range(epochs) :
        model.train()
        y_logits = model(features_train_gpu).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        acc = accuracy(y_true = labels_train_gpu, y_pred=y_pred)
        # now optimize for loss (Train section)
        loss = loss_fn(y_logits,labels_train_gpu)
        optimizer.zero_grad()
        optimizer.step()
        # evaluate on unseen data
        model.eval()
        with torch.inference_mode() :
            _test_logits = model(features_test_gpu).squeeze()
            test_pred = torch.round(torch.sigmoid(_test_logits))
            test_loss = loss_fn(_test_logits,labels_test_gpu)
            test_acc = accuracy(y_true = labels_test_gpu,y_pred = test_pred)
            print(f"Epoch {epoch} Loss {loss} Accuracy {acc} Test Loss {test_loss} Test Acc {test_acc}")
    



    return


@app.cell
def _(X_test, mlutils, model, y_test):
    mlutils.plot_decision_boundary(model=model,features=X_test,labels=y_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
