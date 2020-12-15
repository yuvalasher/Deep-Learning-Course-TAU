from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, roc_curve, mean_squared_error
from consts import MIN_IMPROVEMENT, PATIENT_NUM_EPOCHS
import torch
from torch import nn
from torch.nn.modules import loss


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


def calculate_MSE(y_true: np.array, y_pred: np.array) -> float:
    return mean_squared_error(y_true=y_true.detach().numpy(), y_pred=y_pred.detach().numpy())


def plot_values_by_epochs(train_values: np.array, test_values: np.array, title: str='Loss VS Epoch') -> None:
    """
    Line-plots of 2 sets of values against the epoch value
    """
    fig, ax = plt.subplots()
    ax.plot(list(range(len(train_values))), train_values, label='Train')
    ax.plot(list(range(len(test_values))), test_values, label='Test')
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


def test(net: nn.Module, loss_fn: loss, x_test: np.array, y_test: np.array) -> Tuple[float, np.array]:
    """
    Run the model on x_test and calculate the loss of the predictions.
    The model run on evaluation mode and without updating the computational graph (no_grad)
    """
    net.eval()
    with torch.no_grad():
        y_test_pred = net(x_test.float())
        loss = loss_fn(input=y_test_pred.reshape(-1), target=y_test.float())
        test_loss = loss.item()
    return test_loss, y_test_pred


def check_earlystopping(loss: np.array, epoch: int, min_improvement: float = MIN_IMPROVEMENT,
                        patient_num_epochs: int = PATIENT_NUM_EPOCHS) -> bool:
    """
    Checking convergence in patient_num_epochs before to check if there is still loss improvement in the loss by at
    minimum min_improvement.
    This should be applied on a validation loss, hence, it can cause overfitting on the test set.
    """
    if epoch > patient_num_epochs:
        return np.sum(np.where((loss[epoch - 1 - patient_num_epochs:epoch - 1] -
                                loss[epoch - patient_num_epochs:epoch]) >= min_improvement, 1, 0)) == 0

def plot_decision_boundary(model, x, y):
    # fig, ax = plt.subplots()
    sns.scatterplot(x[:,0], x[:, 1], hue=y)
    keys = list(model.state_dict().keys())
    param1 = model.state_dict()[keys[0]].detach().numpy()
    param2 = model.state_dict()[keys[1]].detach().numpy()
    x1 = np.linspace(x.min()-1, x.max()+1, 10)
    x2 = (-param2[0] - param1[0,0] * x1) / param1[0,1]
    plt.plot(x1, x2, 'r', label="Decision Boundary")
    plt.ylim(top=5)
    plt.ylim(bottom=-5)
    plt.legend()
    plt.show()