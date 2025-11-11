from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device() -> torch.device:
    """
    Returns the appropriate device for our current environment. If GPU can be found it will
    use this.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> Union[float, torch.Tensor]:
    """Calculates classification accuracy.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The classification accuracy as a percentage.
        Returns 0.0 if there are no predictions.
    """
    correct = torch.eq(y_true, y_pred).sum()
    # If y_pred is not empty, calculate accuracy as float percentage
    if len(y_pred) > 0:
        acc: float = (correct.item() / len(y_pred)) * 100
        return acc
    # Otherwise, return 0.0 or the correct count tensor if specific behavior is needed for empty input
    return 0.0


def get_batch_accuracy(output: torch.Tensor, y: torch.Tensor, N: int) -> float:
    """
    Calculate the accuracy of a batch of predictions.

    Args:
        output (torch.Tensor): The output predictions from the model.
        y (torch.Tensor): The true labels.
        N (int): The total number of samples in the batch.

    Returns:
        float: The accuracy of the batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def plot_decision_boundary(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor, figsize=(4, 4)):
    """Plots the decision boundary for a binary classification model.

    Args:
        model: The trained PyTorch model.
        features: A tensor of features (data points), with shape [N, 2].
        labels: A tensor of true labels, with shape [N].
    """
    # Put everything on the same device as the model
    device = next(model.parameters()).device
    model.to(device)
    features = features.to(device)
    labels = labels.to(device)

    # Convert features and labels to numpy for plotting
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Create a meshgrid of points with a small buffer
    # Determine the min/max values for the x and y axes of the plot.
    # A small buffer (0.1) is added to ensure data points aren't exactly on the edge.
    x_min = features_np[:, 0].min() - 0.1
    x_max = features_np[:, 0].max() + 0.1
    y_min = features_np[:, 1].min() - 0.1
    y_max = features_np[:, 1].max() + 0.1

    # Create a mesh grid (xx, yy) of points that covers the entire plot area.
    # This grid represents the space where we will make predictions to draw the boundary.
    # 100 points in each dimension results in 10,000 grid points.

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Combine the mesh grid coordinates into a single array (grid) for prediction.
    # np.c_ stacks them column-wise, and .ravel() flattens them into (10000, 2).
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).type(torch.float).to(device)

    # Make predictions on the grid
    # Set the model to evaluation mode (important for layers like Dropout/BatchNorm).

    model.eval()
    with torch.inference_mode():
        logits = model(grid_tensor).squeeze()
        # Automatically handle binary vs. multi-class
        if len(logits.shape) == 1 or logits.shape[1] == 1:
            # Assumes binary classification with a single output logit
            preds = torch.round(torch.sigmoid(logits.squeeze()))
        else:
            # Assumes multi-class classification
            preds = torch.softmax(logits, dim=1).argmax(dim=1)

    # Reshape the 1D predictions back into the 2D grid shape (100x100) for plotting.
    # Detach the tensor from the computational graph and move it back to the CPU for NumPy/Matplotlib compatibility.
    zz = preds.reshape(xx.shape).detach().cpu().numpy()

    # Plotting

    # Use contourf to fill the decision boundary based on the predictions (zz).
    # alpha=0.2 makes the filled area transparent, allowing data points to be seen.
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(features_np[:, 0], features_np[:, 1], c=labels_np, s=2, cmap=plt.cm.coolwarm)
    # Ensure plot limits match the min/max of the grid created earlier.
    # Ensure plot limits match the min/max of the grid created earlier.
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def in_lab() -> bool:
    """
    Returns True if the code is running in one of the NCCA labs
    Basically we are checking if the hostname is one of the lab machines which begin with either
    pg or w followed by a number. We use a regular expression to do this
    """
    hostname = platform.node()
    # Regular expression pattern
    return re.search(r"(pg|w)\d+", hostname) is not None


def download(url: str, fname: str):
    """
    Downloads a file from the specified url and saves it to the specified file name
        param url: the url of the file to download
        param fname: the file name to save the file as

    """

    resp = requests.get(url, stream=True, verify=False)
    total = int(resp.headers.get("content-length", 0))
    # Can also replace 'file' with a io.BytesIO object
    with (
        open(fname, "wb") as file,
        tqdm(desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024) as progress_bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def unzip_file(zip_file: Path, dest_dir: Path):
    """
    Unzips a file to the specified directory
        param zip_file: the zip file to unzip
        param dest_dir: the directory to unzip the file to
    """

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
