from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, roc_curve
import torch
from typing import Tuple
from sklearn.datasets import make_moons, make_circles

MOONS_NOISE = 0.2


def convert_probs_to_preds(probs: np.array, threshold: float = 0.5, softmax: bool = False) -> np.array:
    """
    Convert probabilities into labels by a given threshold. Probabilities above the threshold will be 1, otherwise 0.
    """
    if softmax:
        return np.argmax(probs, axis=1)
    return np.where(probs >= threshold, 1, 0)


def calculate_accuracy(y_true: np.array, probs: np.array, threshold: float = 0.5) -> float:
    y_pred = convert_probs_to_preds(probs=probs, threshold=threshold)
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def plot_values_by_epochs(train_values: np.array, test_values: np.array, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(list(range(len(train_values))), train_values, label='Train Loss')
    ax.plot(list(range(len(test_values))), test_values, label='Test_loss')
    ax.legend()
    ax.set_ylim(ymin=0)
    ax.set_title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()


def plot_roc_curve(y_test: np.array, y_test_pred: np.array) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    auc_score = auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {auc_score:.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def generate_X_y(size: int, random_num: int) -> Tuple[np.array, np.array]:
    np.random.seed(random_num)
    torch.manual_seed(random_num)
    x, y = make_moons(size, noise=MOONS_NOISE, random_state=random_num)
    return x, y

# # w = weights_initialization(input_dim, 1)
# w = np.random.rand(2, 1)
# b = np.random.randn(1)
# lr = 0.5
#
# def kaiming(m: int, h: int):
#     return np.random.uniform(low=0, high=1, size=(m, h)) * np.sqrt(2. / m)
#
# def weights_initialization(m, h) -> np.array:
#     w = kaiming(m, h)
#     return w
#
# def criterion(y_true: np.array, y_pred: np.array) -> float:
#     return float(np.sum(0.5 * np.power(y_pred - y_true, 2)))
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))
#
# for epoch in tqdm(range(10000)):
#     z = sigmoid(x.dot(w) + b)
#     loss = criterion(y_true=y, y_pred=z)
# #     loss += lambda_reg * np.sum(w ** 2)  # Regularization
#     print(loss)
#     """
#     Update parameters - Backwards
#     dE/dw = dE/dpred * dpred/dz * dz/dw
#     dE/db = dE/dpred * dpred/dz * dz/db (1)
#     """
#     grad_W = ((z - y) * sigmoid_derivative(x)).T.dot(z)
#     grad_b = (z - y) * sigmoid_derivative(z)
#     w -= lr * grad_W
#     for num in (z - y) * sigmoid_derivative(z):
#         b -= lr * num
# #     b += lr * grad_b
