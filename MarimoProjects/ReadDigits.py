import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import mlutils
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import struct

    import torchvision.transforms.v2 as transforms
    return mlutils, nn, np, optim, plt, struct, torch, transforms


@app.cell
def _(mlutils):
    # get our device
    device = mlutils.get_device()
    print(device)
    return (device,)


@app.cell
def _(image, np, struct):
    def load_mnist_labels(filename : str) :
        with open(filename,"rb") as f :
            magic,num = struct.unpack(">II",f.read(8))
            labels = np.fromfile(f,dtype=np.uint8)      
            if len(labels) != num :
                raise ValueError(f"Expected {num} labels got {len(labels)}")
        return labels

    def load_mnist_images(filename : str) :
        with open(filename,"rb") as f :
            magic,num,rows,cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f,dtype=np.uint8).reshape(num,rows,cols)
            if len(images) != num :
                raise ValueError(f"Expected {num} images got {len(image)}")
        return images
    return load_mnist_images, load_mnist_labels


@app.cell
def _(load_mnist_images, load_mnist_labels, mlutils):
    DATASET_LOCATION = ""
    if mlutils.in_lab() :
        DATASET_LOCATION = "/transfer/MNIST/"
    else :
        DATASET_LOCATION ="./MNIST/"


    print(DATASET_LOCATION)

    train_labels = load_mnist_labels(DATASET_LOCATION + "train-labels-idx1-ubyte")
    test_labels = load_mnist_labels(DATASET_LOCATION + "t10k-labels-idx1-ubyte")

    print(len(train_labels),len(test_labels))

    train_images = load_mnist_images(DATASET_LOCATION + "train-images-idx3-ubyte")
    test_images = load_mnist_images(DATASET_LOCATION + "t10k-images-idx3-ubyte")
    print(len(train_images),len(test_images))
    return test_images, test_labels, train_images, train_labels


@app.cell
def _(np, plt, train_images, train_labels):
    def display_image(image: np.array, label: str) -> None:
        plt.figure(figsize=(1, 1))
        plt.title(f"Label : {label}")
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()


    # We can now display the first image from the training dataset.

    display_image(train_images[0], train_labels[0])
    print(type(train_images[0]))
    print(train_images[0].shape)
    print(train_images[0].dtype)
    return (display_image,)


@app.cell
def _(torch, train_images, transforms):
    trans = transforms.Compose(
        [transforms.ToImage(), 
         transforms.ToDtype(torch.float32,scale=True)
        ]
    )
    _new_image = trans(train_images[0])
    print(_new_image.shape)
    print(_new_image.dtype)
    print(_new_image.device)
    print(_new_image.min(),_new_image.max())
    return


@app.cell
def _(device, test_images, test_labels, torch, train_images, train_labels):
    # Convert the data to tensors on the GPU
    train_images_tensor = torch.tensor(train_images,dtype=torch.float32).to(device)
    test_images_tensor = torch.tensor(test_images,dtype=torch.float32).to(device)

    # now the labels these are uint8
    train_labels_tensor = torch.tensor(train_labels,dtype=torch.uint8).to(device)
    test_labels_tensor = torch.tensor(test_labels,dtype=torch.uint8).to(device)




    print(train_images_tensor.shape)
    print(train_images_tensor.device)
    print(train_images_tensor.dtype)
    print(train_images_tensor.min(),train_images_tensor.max())
    return (
        test_images_tensor,
        test_labels_tensor,
        train_images_tensor,
        train_labels_tensor,
    )


@app.cell
def _():
    from torch.utils.data import Dataset,DataLoader
    return DataLoader, Dataset


@app.cell
def _(
    DataLoader,
    Dataset,
    test_images_tensor,
    test_labels_tensor,
    train_images_tensor,
    train_labels_tensor,
):
    class DigitsDataset(Dataset) :
        def __init__(self,images_tensor,labels_tensor) :
            self.images_tensor = images_tensor
            self.labels_tensor = labels_tensor
            assert len(self.images_tensor) == len(self.labels_tensor) 

        def __len__(self) :
            return len(self.images_tensor)

        def __getitem__(self,idx) :
            image = self.images_tensor[idx]
            label = self.labels_tensor[idx]
            return image,label

    train_data = DigitsDataset(train_images_tensor,train_labels_tensor)
    valid_data = DigitsDataset(test_images_tensor,test_labels_tensor)

    batch_size = 32

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data,batch_size=batch_size)

    train_N = len(train_loader.dataset)
    valid_N = len(valid_loader.dataset)
    print(train_N,valid_N)
    return train_N, train_loader, valid_N, valid_loader


@app.cell
def _(device, nn, torch):
    # Build our NN Layers
    n_classes = 10 # digits 0 - 9
    input_size =  28 * 28
    print(input_size)
    layers = [
        nn.Flatten(),
        nn.Linear(input_size,512),
        nn.ReLU(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Linear(128,n_classes)
    ]

    model = nn.Sequential(*layers)
    model.to(device)
    model_compiled = torch.compile(model)
    return input_size, model, model_compiled, n_classes


@app.cell
def _(nn):
    loss_function = nn.CrossEntropyLoss()
    return (loss_function,)


@app.cell
def _(model_compiled, optim):
    optimizer = optim.Adam(model_compiled.parameters())
    return (optimizer,)


@app.function
def get_batch_accuracy(y_true,y_pred,N) :
    pred = y_true.argmax(dim=1, keepdim=True)
    correct = pred.eq(y_pred.view_as(pred)).sum().item()
    return correct / N


@app.cell
def _(loss_function, model_compiled, optimizer, train_N, train_loader):
    def train() :
        loss = 0
        accuracy = 0
        model_compiled.train()
        for x,y in train_loader :
            #x,y  = (x.to(device),y.to(device))
            output =model_compiled(x)
            optimizer.zero_grad()
            batch_loss = loss_function(output,y)
            batch_loss.backward()
            optimizer.step()
            loss = loss + batch_loss.item()
            accuracy = accuracy + get_batch_accuracy(output,y,train_N)
        print(f"Train Loss {loss} Accuracy {accuracy}")
    return (train,)


@app.cell
def _(device, loss_function, model_compiled, torch, valid_N, valid_loader):
    def validate() :
        loss = 0
        accuracy =0
        model_compiled.eval() 
        with torch.no_grad() :
            for x,y in valid_loader :
                x,y = (x.to(device),y.to(device))
                output = model_compiled(x)
                loss = loss + loss_function(output,y).item()
                accuracy = accuracy + get_batch_accuracy(output,y,valid_N)
        print(f"Valid Loss {loss} Accuracy {accuracy}")

    return (validate,)


@app.cell
def _(train, validate):
    epochs = 10

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train()
        validate()

    return


@app.cell
def _(device, display_image, model_compiled, test_images, test_images_tensor):
    index = 9334
    _p = model_compiled(test_images_tensor[index].to(device).unsqueeze(0))
    print(_p)
    print(_p.argmax(dim=1,keepdim=True).item())
    display_image(test_images[index],_p.argmax(dim=1,keepdim=True).item())
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(),"mnist_model.pth")
    return


@app.cell
def _(
    device,
    display_image,
    input_size,
    n_classes,
    nn,
    test_images,
    test_images_tensor,
    torch,
):
    _index=12
    _layers = [
        nn.Flatten(),
        nn.Linear(input_size,512),
        nn.ReLU(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Linear(128,n_classes)
    ]
    model2 = nn.Sequential(*_layers)
    model2.load_state_dict(torch.load("mnist_model.pth"))
    model2.to(device)
    model2.eval()
    prediction = model2(test_images_tensor[_index].to(device).unsqueeze(0))
    display_image(test_images[_index],prediction.argmax(dim=1,keepdim=True).item())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
