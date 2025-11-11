#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification

    In the [previous notebook](SimpleClassification.ipynb) we build a simple Linear model to try and classify a simple circle dataset. We discovered that it could not do this well. We will now try to build a more complex model to see if we can improve the performance.

    ## Getting started

    We are going to start by importing our base libraries and setting the random seed for reproducibility.
    """)
    return


@app.cell
def _():
    import mlutils
    from sklearn.datasets import make_circles
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return make_circles, mlutils, nn, optim, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data generation

    We will use the  ```make_circles``` function from the ```sklearn.datasets``` module to generate our data. This function generates a large circle containing a smaller circle in 2D. A simple toy dataset to visualize clustering and classification algorithms.
    """)
    return


@app.cell
def _(make_circles):
    # Make 1000 samples
    n_samples = 2000

    # Create circles
    x, y = make_circles(
        n_samples,
        noise=0.03,
        # random_state=42,  # a little bit of noise to the dots
    )  # keep random state so we get the same values
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see the dataset
    """)
    return


@app.cell
def _(plt, x, y):
    plt.figure(figsize=(4, 4))

    plt.scatter(x=x[:, 0], y=x[:, 1], c=y, s=2, cmap=plt.cm.plasma)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing

    We now need to convert our data into tensors and generate our test / train split. We will use the typical 80:20 split for this.
    """)
    return


@app.cell
def _(torch, x, y):
    features = torch.from_numpy(x).type(torch.float)
    labels = torch.from_numpy(y).type(torch.float)
    train_split = int(0.8 * len(features))
    features_train = features[:train_split]
    features_test = features[train_split:]
    labels_train = labels[:train_split]
    labels_test = labels[train_split:]
    print(
        features_train.shape,
        features_test.shape,
        labels_train.shape,
        labels_test.shape,
    )
    return features_test, features_train, labels, labels_test, labels_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building a Model

    In the previous example we tried to use a line to fit data that is not linear. We will now try to use a non linear activation function to see if we can fit our data better.

    PyTorch has a number of [non linear activation](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) functions we can use.

    In this example we will use the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function defined in  [`torch.nn.ReLU()`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)).
    """)
    return


@app.cell
def _(mlutils):
    # Make device agnostic code
    device = mlutils.get_device()
    print(device)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will still inherit from the nn.Module but change the shape of our model to include a hidden layer with a ReLU activation function.
    """)
    return


@app.cell
def _(device, nn):
    # Build model with non-linear activation function
    class BinaryClassify(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10)
            self.layer_3 = nn.Linear(in_features=10, out_features=1)
            self.relu = nn.ReLU()  # non-linear activation function

        def forward(self, x):
            # This is a more complex model with 3 layers and a
            # non-linear activation function between each layer
            return self.layer_3(
                self.relu(self.layer_2(self.relu(self.layer_1(x))))
            )


    model = BinaryClassify().to(device)
    print(model.parameters)
    print(model.state_dict())
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can now see this is a far more complex model than the previous one. If you uncomment the .state_dict() line you can see the weights and biases of the model and there are many more than before.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use this model (basically forward is written for us now) this is really easy and for simple tasks ideal. However for more complex models we will need to use the more manual method and define our own forward function.

    We can now see what our untrained model does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Function and Optimizer

    As we are dealing with a binary classification problem we can choose a suitable function from the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#loss-functions).

    PyTorch has two
    1. [`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    2. [`torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

    They are both similar however the BCEWithLogitsLoss is more numerically stable and has a built in sigmoid function.

    We will use the Stochastic Gradient Descent (SGD) optimizer to train our model as in the previous lab.
    """)
    return


@app.cell
def _(model, nn, optim):
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(params=model.parameters(), lr=0.1)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we need to see how close our function is to the actual labels we can generate a function to do this this is know as an evaluation function and is basically the opposite of the loss function but it can sometimes be more useful to see how well the model is doing.

    In the previous example we had an accuracy function, this has now been added to the MLUtils module as we will use this a lot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the Model

    We will now train the model using the training data. We will use the same training loop as in the previous lab but using the new data sets, we will now also copy the data to the device to help speed up the training process.
    """)
    return


@app.cell
def _(
    device,
    features_test,
    features_train,
    labels_test,
    labels_train,
    loss_fn,
    mlutils,
    model,
    optimizer,
    torch,
):
    torch.manual_seed(42)
    epochs = 500

    features_train_gpu = features_train.to(device)
    features_test_gpu = features_test.to(device)
    labels_train_gpu = labels_train.to(device)
    labels_test_gpu = labels_test.to(device)


    for epoch in range(epochs):
        model.train()
        y_logits = model(features_train_gpu).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, labels_train_gpu)
        acc = mlutils.accuracy(y_true=labels_train_gpu, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            _test_logits = model(features_test_gpu).squeeze()
            test_pred = torch.round(torch.sigmoid(_test_logits))
            test_loss = loss_fn(_test_logits, labels_test_gpu)
            test_acc = mlutils.accuracy(y_true=labels_test_gpu, y_pred=test_pred)
        if epoch % 20 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
            )
    return (features_test_gpu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is much better accuracy around 80% which is much better than the previous model. Let's make some predictions and see how well the model is doing.
    """)
    return


@app.cell
def _(features_test_gpu, labels, model, torch):
    model.eval()
    with torch.inference_mode():
        y_preds = torch.round(torch.sigmoid(model(features_test_gpu))).squeeze()
    print(y_preds[:10], labels[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the same code we used in the previous lab to plot the decision boundary.
    """)
    return


@app.cell
def _(features_test, labels_test, mlutils, model):
    mlutils.plot_decision_boundary(
        model=model, features=features_test.cpu(), labels=labels_test.cpu()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is much better, you can see for the most part the model has correctly classified the data. With a little more training and a few tweaks we could probably get this to >90% accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
