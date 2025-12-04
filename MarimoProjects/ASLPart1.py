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
    return (
        Adam,
        DataLoader,
        Dataset,
        mlutils,
        nn,
        pathlib,
        pd,
        plt,
        string,
        torch,
        zipfile,
    )


@app.cell
def _(mlutils):
    device = mlutils.get_device()
    print(device)
    return (device,)


@app.cell
def _(mlutils, pathlib):
    # going to download the ASL dataset to the /transfer or local drive if /transfer not there.
    if mlutils.in_lab() :
        DATASET_LOCATION="/transfer/mnist_asl"
    else :
        DATASET_LOCATION="./mnist_asl"
    print(f"using {DATASET_LOCATION}")
    pathlib.Path(DATASET_LOCATION).mkdir(parents=True,exist_ok=True)
    return (DATASET_LOCATION,)


@app.cell
def _(DATASET_LOCATION, mlutils, pathlib, zipfile):
    URL = "https://www.kaggle.com/api/v1/datasets/download/nadaemad2002/asl-mnist"

    dest = DATASET_LOCATION+"/asl-mnist.zip"
    if not pathlib.Path(dest).exists() :
        mlutils.download(URL,dest)
        with zipfile.ZipFile(dest,"r") as zip_ref :
            zip_ref.extractall(DATASET_LOCATION)

    return


@app.cell
def _(DATASET_LOCATION, pathlib):
    for file in pathlib.Path(DATASET_LOCATION).glob("*") :
        print(file)
    return


@app.cell
def _(DATASET_LOCATION, pd):
    train_df = pd.read_csv(f"{DATASET_LOCATION}/sign_mnist_train.csv")
    test_df = pd.read_csv(f"{DATASET_LOCATION}/sign_mnist_test.csv")
    return test_df, train_df


@app.cell
def _(test_df):
    test_df.head()
    return


@app.cell
def _(test_df, train_df):
    y_train = train_df.pop("label")
    y_valid = test_df.pop("label")
    print(y_train.value_counts(sort=True,ascending=True).sort_index())
    return y_train, y_valid


@app.cell
def _(y_train):
    print(y_train.agg(["min","max"]))
    return


@app.cell
def _(test_df, train_df):
    x_train=train_df.values
    x_valid = test_df.values
    print(x_train.shape,x_valid.shape)
    print(type(x_train),type(x_valid))
    return (x_train,)


@app.cell
def _():
    return


@app.cell
def _(plt, string, x_train, y_train):
    def plot_image(images,labels,num_images,image_index) :
        image=images.reshape(28,28)
        label = labels
        plt.subplot(1,num_images,image_index+1)
        plt.title(label,fontdict={"fontsize" : 30})
        plt.axis("off")
        plt.imshow(image,cmap="gray")

    alphabet = string.ascii_letters[:25]
    num_images=10
    start_image = 12000
    plt.figure(figsize=(10,10))
    for index,x in enumerate(range(start_image,start_image+num_images)) :
        row = x_train[x]
        label = y_train[x]
        plot_image(row,alphabet[label],num_images,index)
    plt.show()
    return


@app.cell
def _(test_df, train_df):
    # NN needs our data in float format, this is in int 0/255
    images_train = train_df.values / 255
    images_test = test_df.values / 255
    return images_test, images_train


@app.cell
def _(Dataset, device, images_test, images_train, torch, y_train, y_valid):
    class ASLDataSet(Dataset) :
        def __init__(self,x_df,y_df) :
            self.xs = torch.tensor(x_df).float().to(device)
            self.ys = torch.tensor(y_df).to(device)
            assert len(self.xs) == len(self.ys)
        def __getitem__(self,idx) :
            x = self.xs[idx]
            y= self.ys[idx]
            return x,y

        def __len__(self) :
            return len(self.xs)

        def save(self,path) :
            data = {
                "xs" : self.xs.cpu(),
                "ys" : self.ys.cpu()
            }
            torch.save(data,path)

    train_data = ASLDataSet(images_train,y_train)
    valid_data = ASLDataSet(images_test,y_valid)

    train_data.save("train_data.pth")
    valid_data.save("valid_data.pth")
    return train_data, valid_data


@app.cell
def _(DataLoader, train_data, valid_data):
    BATCH_SIZE = 32
    train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(valid_data,batch_size=BATCH_SIZE)
    return test_loader, train_loader


@app.cell
def _(train_data):
    input_size=len(train_data[0][0]) # 28 * 28 pixels of the image
    num_classes = 25 # a - y
    return input_size, num_classes


@app.cell
def _(device, input_size, nn, num_classes, torch):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size,512), # input layer
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,num_classes)
    )

    model_compiled=torch.compile(model.to(device))
    model_compiled.to(device)
    return (model_compiled,)


@app.cell
def _(Adam, model_compiled, nn):
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model_compiled.parameters())
    return loss_function, optimizer


@app.cell
def _(
    loss_function,
    mlutils,
    model_compiled,
    optimizer,
    test_loader,
    train_loader,
):
    train_accuracy = []
    train_loss = []
    train_N = len(train_loader.dataset)
    valid_N = len(test_loader.dataset)

    def train() :
        loss =0
        accuracy = 0
        model_compiled.train()
        for x,y in train_loader :
            output = model_compiled(x)
            optimizer.zero_grad()
            batch_loss = loss_function(output,y)
            batch_loss.backward()
            optimizer.step()
            loss+=batch_loss.item()
            accuracy+= mlutils.get_batch_accuracy(output,y,train_N) 
        train_accuracy.append(accuracy)
        train_loss.append(loss)
        print(f"Train Loss {loss:.4f} Accuracy {accuracy:.4f}")
    return train, train_accuracy, train_loss, valid_N


@app.cell
def _(loss_function, mlutils, model_compiled, test_loader, torch, valid_N):
    valid_accuracy = []
    valid_loss = []

    def validate() :
        loss = 0
        accuracy =0
        model_compiled.eval()
        with torch.no_grad() :
            for x,y in test_loader :
                output = model_compiled(x)
                loss += loss_function(output,y).item()
                accuracy += mlutils.get_batch_accuracy(output,y,valid_N)
        valid_accuracy.append(accuracy)
        valid_loss.append(loss)
        print(f"Valid Loss {loss:.4f}  Accuracy {accuracy:.4f}")

    return valid_accuracy, valid_loss, validate


@app.cell
def _(train, validate):
    epochs = 10
    for epoch in range(epochs) :
        print(f"{epoch=}")
        train()
        validate()
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


if __name__ == "__main__":
    app.run()
