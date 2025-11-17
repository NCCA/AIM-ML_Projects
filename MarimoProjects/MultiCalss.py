import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import mlutils
    from sklearn.datasets import make_blobs
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return make_blobs, mlutils, nn, optim, plt, torch


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
    ## Multi classification

    We will generate a set of data which we can use to identify areas in an bigger image / layout.
    """)
    return


@app.cell
def _(make_blobs, plt):
    n_samples = 1000
    CLASSES = 4
    FEATURES = 2
    SEED = 12345
    input_features,input_labels = make_blobs(n_samples=n_samples, n_features = FEATURES,centers = CLASSES, cluster_std=0.9, random_state=SEED)
    plt.figure(figsize=(4,4))
    plt.scatter(x=input_features[:,0],y=input_features[:,1], c=input_labels)
    plt.show()

    return CLASSES, FEATURES, input_features, input_labels


@app.cell
def _(input_features, input_labels, torch):
    features = torch.from_numpy(input_features).type(torch.float)
    labels = torch.from_numpy(input_labels).type(torch.float)
    train_split = int(0.8 * len(features))
    features_train = features[:train_split]
    features_test = features[train_split:]
    labels_train = labels[:train_split]
    labels_test = labels[train_split:]

    print(features_train.shape,features_test.shape)
    print(labels_train.shape,labels_test.shape)
    print(labels_train.dtype)

    return features_test, features_train, labels, labels_test, labels_train


@app.cell
def _(mlutils):
    device = mlutils.get_device()
    print(device)
    return (device,)


@app.cell
def _(CLASSES, FEATURES, device, nn):
    # build a non-linear classify NN model

    class Classify(nn.Module) :
        def __init__(self,input_features, output_features, hidden_size=8) :
            super().__init__()
            self.linear_layer = nn.Sequential(
                nn.Linear(in_features=input_features, out_features = hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size,out_features = hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size,out_features=output_features)
            )
        def forward(self,x) :
            return self.linear_layer(x)


    model = Classify(input_features=FEATURES,output_features=CLASSES).to(device)
    print(model.parameters)


        
    return (model,)


@app.cell
def _(model, nn, optim):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),lr=0.1)
    return loss_fn, optimizer


@app.cell
def _(device, features_train, model, torch):
    _logits = model(features_train.to(device))
    print(_logits.shape)
    print(_logits[:5])

    _probabilities = torch.softmax(_logits,dim=1)
    print(_probabilities[:5])

    return


@app.cell
def _(device, features_test, features_train, labels_test, labels_train, torch):
    torch.manual_seed(1234)
    epochs = 100 

    features_train_gpu = features_train.to(device)
    features_test_gpu = features_test.to(device)
    # convert our data to long for the labels
    labels_train_gpu = labels_train.type(torch.LongTensor).to(device)
    labels_test_gpu = labels_test.type(torch.LongTensor).to(device)

    return (
        epochs,
        features_test_gpu,
        features_train_gpu,
        labels_test_gpu,
        labels_train_gpu,
    )


@app.cell
def _(
    epochs,
    features_test_gpu,
    features_train_gpu,
    labels_test_gpu,
    labels_train_gpu,
    loss_fn,
    mlutils,
    model,
    optimizer,
    torch,
):
    for epoch in range(epochs) :
        ## train 
        model.train()
        # forward pass
        y_logits = model(features_train_gpu)
        # turn logits (- +) into probabilites
        y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
        # calculate loss / accruacy
        loss = loss_fn(y_logits,labels_train_gpu)
        acc = mlutils.accuracy(y_true=labels_train_gpu,y_pred=y_pred)
        # reset optimizer
        optimizer.zero_grad()
        # calculate gradiants
        loss.backward()
        # update weights
        optimizer.step()

        # test
        model.eval()
        with torch.inference_mode() :
            # forward pass 
            test_logits = model(features_test_gpu)
            test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)
            # test loss
            test_loss = loss_fn(test_logits,labels_test_gpu)
            test_acc = mlutils.accuracy(y_true=labels_test_gpu,y_pred=test_pred)
            if epoch % 10 == 0 :
                print(f"Epoch Train | {epoch} Loss {loss} Accuracy {acc} | Test {test_loss} Accuracy {test_acc}")

    
    return


@app.cell
def _(
    features_test,
    features_test_gpu,
    labels,
    labels_test,
    mlutils,
    model,
    torch,
):
    model.eval()
    with torch.inference_mode() :
        y_preds = torch.round(torch.sigmoid(model(features_test_gpu))).squeeze()
    print(y_preds[:10],labels[:10])
    mlutils.plot_decision_boundary(
        model=model, features=features_test.cpu(), labels=labels_test.cpu()
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
