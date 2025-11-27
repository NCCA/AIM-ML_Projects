import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import Dataset,DataLoader

    # vision transforms
    import torchvision
    import torchvision.transforms.v2 as transforms
    import torchvision.transforms.functional as F

    import mlutils
    device = mlutils.get_device()
    print(device)
    return Adam, DataLoader, device, nn, torch, torchvision, transforms


@app.cell
def _():
    DATASET_LOCATION = "/transfer/mnist_data/"
    return (DATASET_LOCATION,)


@app.cell
def _(DATASET_LOCATION, torchvision):
    train_set = torchvision.datasets.MNIST(DATASET_LOCATION,train=True,download=True)
    valid_set = torchvision.datasets.MNIST(DATASET_LOCATION,train=False,download=True)

    return train_set, valid_set


@app.cell
def _(train_set, valid_set):
    print(type(train_set))
    print(train_set)
    print(valid_set)
    return


@app.cell
def _(train_set):
    x_0,y_0 = train_set[0]
    print(type(x_0),type(y_0))
    x_0
    return


@app.cell
def _(torch, train_set, transforms, valid_set):
    trans = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32,scale=True)
    ])
    train_set.transform = trans
    valid_set.transform = trans
    _x,_y = train_set[0]

    return


@app.cell
def _(DataLoader, train_set, valid_set):
    batch_size = 32
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=batch_size)

    return train_loader, valid_loader


@app.cell
def _(device, nn, torch):
    n_classes = 10
    input_size = 28 * 28
    layers = [
        nn.Flatten(),
        nn.Linear(input_size,512), # input layer
        nn.ReLU(), # activation for input
        nn.Linear(512,512), # hidden layer 
        nn.ReLU(), # activatio for output 
        nn.Linear(512,n_classes) # output layer
    ]

    model = nn.Sequential(*layers)
    model.to(device)
    model=torch.compile(model)
    return (model,)


@app.cell
def _(Adam, model, nn):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    return loss_fn, optimizer


@app.cell
def _(train_loader, valid_loader):
    train_N = len(train_loader.dataset)
    valid_N = len(valid_loader.dataset)
    print(train_N,valid_N)
    return train_N, valid_N


@app.function
def get_batch_accuracy(output,y,N) :
    pred = output.argmax(dim=1,keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


@app.cell
def _(device, loss_fn, model, optimizer, train_N, train_loader):
    def train() :
        loss =0
        accuracy = 0
        model.train()
        for x,y in train_loader :
            x=x.to(device)
            y=y.to(device)
            output = model(x)
            optimizer.zero_grad()
            batch_loss = loss_fn(output,y)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            accuracy += get_batch_accuracy(output,y,train_N)
        print(f"Train - Loss {loss:.4f} Accuracy {accuracy:.4f}")
        
    return (train,)


@app.cell
def _(device, loss_fn, model, torch, valid_N, valid_loader):
    def valid() :
        loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad() :
            for x,y in valid_loader :
                x=x.to(device)
                y=y.to(device)
                output = model(x)
                loss += loss_fn(output,y).item()
                accuracy+= get_batch_accuracy(output,y,valid_N)
        print(f"Valid Loss {loss:.4f} Accuracy {accuracy:.4f}")
        
    return (valid,)


@app.cell
def _(train, valid):
    epochs = 5
    for epoch in range(epochs) :
        print(epoch)
        train()
        valid()
    return


@app.cell
def _(device, model, train_set):
    prediction = model(train_set[0][0].to(device).unsqueeze(0))
    print(prediction.argmax(dim=1, keepdim=True))
    train_set[0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
