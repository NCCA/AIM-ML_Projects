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
    return ASLDataSet, Adam, DataLoader, mlutils, nn, plt, string, torch


@app.cell
def _(mlutils):
    device = mlutils.get_device()
    return (device,)


@app.cell
def _(ASLDataSet, DataLoader):
    train_data = ASLDataSet.load("train_data.pth")
    valid_data = ASLDataSet.load("valid_data.pth")
    # we need an image this time so re-shape to 2D 
    train_data.xs=train_data.xs.reshape(-1,1,28,28)
    valid_data.xs=valid_data.xs.reshape(-1,1,28,28)

    BATCH_SIZE = 32
    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    valid_loader=DataLoader(valid_data,batch_size=BATCH_SIZE)
    input_size = len(train_data[0][0])
    num_classes = 25 # a-y
    print(input_size)
    kernel_size = 3
    flattened_image_size = 75 * 3 * 3

    print(train_data.xs[0].shape)
    return (
        flattened_image_size,
        kernel_size,
        num_classes,
        train_loader,
        valid_loader,
    )


@app.cell
def _(flattened_image_size, kernel_size, nn, num_classes):
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNELS = 1
    model = nn.Sequential(
        # First Convoloution
        nn.Conv2d(IMAGE_CHANNELS,25,kernel_size,stride=1,padding=1),  # 25 x 28 x 28 
        nn.BatchNorm2d(25),
        nn.ReLU(),
        nn.MaxPool2d(2,stride=2), # 25 x 14 x 14
        # 2nd convolution
        nn.Conv2d(25,50,kernel_size,stride=1, padding=1),
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2,stride=2), # 50 x 7 x 7
        # 3rd Convolution
        nn.Conv2d(50,75,kernel_size,stride=1,padding=1), # 75 x 7 x7
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2,stride=2), # 75 x 3 x 3
        # flatten to dense
        nn.Flatten(),
        nn.Linear(flattened_image_size,512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512,num_classes)
    
    )
    return (model,)


@app.cell
def _(device, model, torch):
    model_compiled = torch.compile(model.to(device))
    model_compiled.to(device)

    return (model_compiled,)


@app.cell
def _(Adam, model_compiled, nn):
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model_compiled.parameters(), lr = 0.0001)
    return loss_function, optimizer


@app.cell
def _(train_loader, valid_loader):
    train_accuracy = []
    train_loss =[] 
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
    epochs = 10 
    for epoch in range(epochs) : 
        print(f"{epoch=}")
        loss,accuracy = mlutils.train(model_compiled,train_loader,optimizer,loss_function,device,train_N)
        print(f"Train loss {loss:.02f} accuracy {accuracy:0.4f}")
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        ## validate
        loss,accuracy = mlutils.validate(model_compiled,valid_loader,loss_function,device,valid_N)
        print(f"Valid loss {loss:.02f} accuracy {accuracy:0.4f}")
        valid_loss.append(loss)
        valid_accuracy.append(accuracy)
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
def _(device, model_compiled, plt, string, torch, train_loader):
    alphabet = string.ascii_letters[:]
    model_compiled.eval()
    with torch.no_grad() :
        x,y = next(iter(train_loader))
        print(x.shape)
        output=model_compiled(x.to(device))
    
        pred=output.argmax(dim=1, keepdim=True)
        num_images = 32
        plt.figure(figsize=(20,20))
        for _i in range(num_images) :
            plt.subplot(1,num_images,_i+1)
            plt.title(f"{alphabet[y[_i].item()]}  {alphabet[pred[_i].item()]} ")
            plt.axis("off")
            plt.imshow(x[_i].cpu().numpy().reshape(28,28),cmap="gray")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
