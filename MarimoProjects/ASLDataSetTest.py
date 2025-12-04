import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
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
    return ASLDataSet, Adam, DataLoader, mlutils, nn, plt, torch


@app.cell
def _(mlutils):
    device = mlutils.get_device()
    return (device,)


@app.cell
def _(ASLDataSet, DataLoader):
    train_data = ASLDataSet.load("train_data.pth")
    valid_data = ASLDataSet.load("valid_data.pth")

    BATCH_SIZE = 32
    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    valid_loader=DataLoader(valid_data,batch_size=BATCH_SIZE)


    return train_data, train_loader, valid_loader


@app.cell
def _(train_data):
    input_size = len(train_data[0][0])
    num_classes = 25 # a-y
    print(input_size)
    return input_size, num_classes


@app.cell
def _(device, input_size, nn, num_classes, torch):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size,512),
        nn.ReLU(), 
        nn.Linear(512,512),
        nn.ReLU(), 
        nn.Linear(512,num_classes)
    )
    model_compiled = torch.compile(model.to(device))
    model_compiled.to(device)
    return model, model_compiled


@app.cell
def _(Adam, model_compiled, nn):
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model_compiled.parameters())
    return loss_function, optimizer


@app.cell
def _(train_loader, valid_loader):
    train_accuracy = []
    train_loss = []

    valid_accuracy = []
    valid_loss = []

    train_N = len(train_loader.dataset)
    valid_N = len(valid_loader.dataset)
    return (
        train_N,
        train_accuracy,
        train_loss,
        valid_N,
        valid_accuracy,
        valid_loss,
    )


@app.cell
def _(
    device,
    loss_function,
    mlutils,
    model,
    model_compiled,
    optimizer,
    train_N,
    train_accuracy,
    train_loader,
    train_loss,
    valid_N,
    valid_accuracy,
    valid_loader,
    valid_loss,
):
    for epoch in range(20) :
        loss,accuracy = mlutils.train(model_compiled,train_loader,optimizer,loss_function,device,train_N)
        print(f"{epoch=} Train Loss {loss} accuracy {accuracy}")
        train_loss.append(loss)
        train_accuracy.append(accuracy)

        loss,accuracy = mlutils.validate(model,valid_loader,loss_function,device,valid_N)
        valid_loss.append(loss)
        valid_accuracy.append(accuracy)
        print(f"{epoch=} Test {loss} Accuracy {accuracy}")
    return


@app.cell
def _(plt, train_accuracy, train_loss, valid_accuracy, valid_loss):
    plt.figure(figsize=(5,2))
    plt.plot(train_accuracy, label="Train")
    plt.plot(valid_accuracy, label="Valid")
    plt.xlabel("Epoch")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,2))
    plt.plot(train_loss, label="Train")
    plt.plot(valid_loss, label="Valid")
    plt.xlabel("Epoch")
    plt.title("Loss")
    plt.legend()
    plt.show()

    return


@app.cell
def _():



    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
