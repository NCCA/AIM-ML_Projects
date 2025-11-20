import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import mlutils
    from pathlib import Path
    DATASET_LOCATION = ""
    if mlutils.in_lab() :
        DATASET_LOCATION = "/transfer/MNIST/"
    else :
        DATASET_LOCATION ="./MNIST/"

    print(DATASET_LOCATION)
    Path(DATASET_LOCATION).mkdir(parents=True,exist_ok=True)

    p=Path("/transfer")
    for child in p.iterdir() :
        print(child)
    return DATASET_LOCATION, Path, mlutils


@app.cell
def _(DATASET_LOCATION, Path, mlutils):

    files=[
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    URL="https://storage.googleapis.com/cvdf-datasets/mnist/"
    for _file in files :
        if not Path(DATASET_LOCATION + _file).exists() :
            print(f"Downloading {_file}")
            mlutils.download(f"{URL}{_file}",f"{DATASET_LOCATION}/{_file}")
    return (files,)


@app.cell
def _(DATASET_LOCATION, Path, files):
    import gzip
    for _file  in files :
        if not Path(DATASET_LOCATION+_file[:-3]).exists() :
            print(f"Unziping {_file}")
            with gzip.open(DATASET_LOCATION + _file,"rb") as f_in :
                with open(DATASET_LOCATION + _file[:-3], "wb") as f_out :
                    f_out.write(f_in.read())
    return


@app.cell
def _(DATASET_LOCATION):
    import numpy as np
    import matplotlib.pyplot as plt
    import struct

    def load_mnist_labels(filename : str) :
        with open(filename,"rb") as f :
            magic,num = struct.unpack(">II",f.read(8))
            labels = np.fromfile(f,dtype=np.uint8)      
            if len(labels) != num :
                raise ValueError(f"Expected {num} labels got {len(labels)}")
        return labels

    train_labels = load_mnist_labels(DATASET_LOCATION + "train-labels-idx1-ubyte")
    test_labels = load_mnist_labels(DATASET_LOCATION + "t10k-labels-idx1-ubyte")
    print(len(train_labels),len(test_labels))
    print(train_labels[:100])
    return np, plt, struct, test_labels, train_labels


@app.cell
def _(DATASET_LOCATION, image, np, struct):
    def load_mnist_images(filename : str) :
        with open(filename,"rb") as f :
            magic,num,rows,cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f,dtype=np.uint8).reshape(num,rows,cols)
            if len(images) != num :
                raise ValueError(f"Expected {num} images got {len(image)}")
        return images

    train_images = load_mnist_images(DATASET_LOCATION+"train-images-idx3-ubyte")
    test_images = load_mnist_images(DATASET_LOCATION+"t10k-images-idx3-ubyte")

    print(train_images.shape)
    return test_images, train_images


@app.cell
def _(plt, test_images, test_labels, train_images, train_labels):
    def display_image(image,label) :
        plt.title(f"Label {label}")
        plt.imshow(image,cmap="gray")
        plt.show()

    display_image(train_images[0],train_labels[0])
    display_image(test_images[0],test_labels[0])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
