---
title: 03Multiclassification
marimo-version: 0.17.7
width: full
header: |-
  #!/usr/bin/env -S uv run marimo edit
---

# Multi Label Classification

In the [previous notebook](BinaryClassification.ipynb) we build a model to do binary classification. In this notebook we are going to build a model to do multi label classification.

## Getting started

We are going to start by importing our base libraries and setting the random seed for reproducibility.

```python {.marimo}
import mlutils
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
```

## Data generation

We will use the  ```make_blobs``` function from the ```sklearn.datasets``` module to generate our data.

```python {.marimo}
# Make 1000 samples
n_samples = 1000

CLASSES = 4
FEATURES = 2
SEED = 12345

# 1. Create multi-class data
x, y = make_blobs(
    n_samples=n_samples,
    n_features=FEATURES,
    centers=CLASSES,
    cluster_std=1.2,
    random_state=SEED,
)
```

Let's see the dataset

```python {.marimo}
plt.figure(figsize=(4, 4))

plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.plasma)
plt.show()
```

## Data Preprocessing

We now need to convert our data into tensors and generate our test / train split. We will use the typical 80:20 split for this.

```python {.marimo}
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
```

## Building a Model

We are going to setup our model in a similar way as before, however this time we have more than one output. To make  it easier to write our forward pass we are going to use the ```torch.nn.Sequential``` class which will call the forward method of each module in the order they are passed to the constructor.

```python {.marimo}
# Make device agnostic code
device = mlutils.get_device()
print(device)
```

We will still inherit from the nn.Module but change the shape of our model to include a hidden layer with a ReLU activation function.

```python {.marimo}
# Build model with non-linear activation function


class Classify(nn.Module):
    def __init__(self, input_features=2, output_features=1, hidden_size=8):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer(x)


model = Classify(input_features=FEATURES, output_features=CLASSES).to(device)
print(model.parameters)
```

## Loss Function and Optimizer

This time as we are dealing with more than one output we need to use the CrossEntropyLoss function. This is a combination of the softmax activation function and the negative log likelihood loss function.

```python {.marimo}
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=model.parameters(), lr=0.1)
```

We can now look at the basic output of the untrained model.

```python {.marimo}
logits = model(features_train.to(device))
print(logits.shape)
probabilities = torch.softmax(logits, dim=1)
print(logits[:5])
print(probabilities[:5])
```

The softmax function is used to convert the output of the model into a probability distribution, this should sum to 1. We can then find the class with the highest probability using the argmax function to determine the predicted class.

```python {.marimo}
print(probabilities[0])
print(torch.argmax(probabilities[0]))
```

## Training the Model

We will now train the model using the training data. We will use the same training loop as in the previous lab but using the new data sets, we will now also copy the data to the device to help speed up the training process.

```python {.marimo}
torch.manual_seed(1234)

epochs = 100

# copy to device
features_train_device = features_train.to(device)

# Note we need to convert here as the cuda model doesn't work on floats
labels_train_device = labels_train.type(torch.LongTensor)
labels_train_device = labels_train_device.to(device)

features_test_device = features_test.to(device)
labels_test_device = labels_test.type(torch.LongTensor)
labels_test_device = labels_test_device.to(device)

for epoch in range(epochs):
    ### Training
    model.train()

    # 1. Forward pass
    y_logits = model(features_train_device)
    # turn logits -> pred probs -> pred labls
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, labels_train_device)
    acc = mlutils.accuracy(y_true=labels_train_device, y_pred=y_pred)

    # reset the optimizer to zero
    optimizer.zero_grad()
    # calculate the gradients
    loss.backward()
    # update the weights
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(features_test_device)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        # 2. calculate loss/accuracy
        test_loss = loss_fn(test_logits, labels_test_device)
        test_acc = mlutils.accuracy(
            y_true=labels_test_device, y_pred=test_pred
        )

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )
```

This is much better accuracy around 80% which is much better than the previous model. Let's make some predictions and see how well the model is doing.

```python {.marimo}
model.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model(features_test_device))).squeeze()
print(y_preds[:10], labels[:10])
```

We can use the same code we used in the previous lab to plot the decision boundary.

```python {.marimo}
mlutils.plot_decision_boundary(
    model=model, features=features_test.cpu(), labels=labels_test.cpu()
)
```

This works well. Try re-running the model with the ReLU activation function removed and see how this affects the decision boundary.

```python {.marimo hide_code="true"}
mo.md(r"""

""")
```

```python {.marimo}
import marimo as mo
```
