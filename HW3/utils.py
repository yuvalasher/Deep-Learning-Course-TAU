import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, auc, accuracy_score, auc, \
    roc_curve, mean_squared_error

from consts import *
import os
import random
import h5py
from typing import Any


def _activate_random_seed(random_seed: int = SEED) -> None:
    """
    Activating the randomness of torch & numpy by defined random seed
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)


def calculate_model_metrics(y_true: np.array, y_pred: np.array, verbose: bool = True, mode: str = 'Test') -> Tuple[
    float, float, float]:
    """
    Calculating Accuracy, recall, precision
    """
    if type(y_true) == torch.tensor:
        y_true = y_true.detach().numpy()
    y_pred = convert_probs_to_preds(probs=y_pred)
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    assert y_true.shape == y_pred.shape
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)  # , average='samples')
    precision = precision_score(y_true=y_true, y_pred=y_pred)  # , average=='samples')
    if verbose:
        print(f'*** {mode} ***')
        print(f'Num of found targets: {(y_pred[np.where(y_true == 1)[0]] == 1).sum()} / {int(y_true.sum())}')
        print(f'Accuracy: {accuracy * 100:.2f}%; Recall: {recall * 100:.2f}%; Precision: {precision * 100:.2f}%')

    return accuracy, recall, precision


def convert_probs_to_preds(probs: np.array, threshold: float = 0.5) -> np.array:
    """
    Convert probabilities into labels by a given threshold. Probabilities above the threshold will be 1, otherwise 0.
    """
    return np.where(probs >= threshold, 0, 1)


def calculate_accuracy(y_true: np.array, probs: np.array, threshold: float = 0.5) -> float:
    y_pred = convert_probs_to_preds(probs=probs, threshold=threshold)
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def calculate_num_corrects(y_true: np.array, y_pred: np.array, threshold: float = 0.5):
    y_pred = convert_probs_to_preds(probs=y_pred, threshold=threshold)
    return (y_true == y_pred).sum()


def calculate_MSE(y_true: np.array, y_pred: np.array) -> float:
    return mean_squared_error(y_true=y_true.detach().numpy(), y_pred=y_pred.detach().numpy())


def plot_values_by_epochs(train_values: np.array, test_values: np.array, title: str = 'Loss VS Epoch') -> None:
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


def calculate_auc_score(y_true: np.array, y_pred: np.array) -> float:
    if type(y_true) == torch.Tensor:
        y_true = y_true.detach().numpy()
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.detach().numpy()

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    return auc(fpr, tpr)


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


def save_hdf5_to_file(obj_name: str, obj: object) -> None:
    with h5py.File(PROJECT_PATH / 'hdf5' / f'{obj_name}.h5', 'w') as hf:
        hf.create_dataset(obj_name, data=obj)


def load_hdf5_file(obj_name: str) -> Any:
    with h5py.File(PROJECT_PATH / 'hdf5' / f'{obj_name}.h5', 'r') as hf:
        return hf[obj_name][:]


def print_trainable_params(net: nn.Module) -> None:
    """
    Printing all the trainable parameters in the net - all the parameters with requires_grad = True
    (will create a computational graph)
    """
    print(net)
    print('Trainable Params:')
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print('\t', name)


def get_num_users_items(df: pd.DataFrame) -> Tuple[int, int]:
    return df.UserID.max() + 1, df.ItemID.max() + 1


def get_train_random_validation_data(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creating random validation set from the train set - for each user, there is only one record in the validation
    The item the user liked is in the first column ("Item1")
    """
    num_users, num_items = get_num_users_items(df=train_df)
    validation_indices = []
    validation_df = pd.DataFrame(columns=['UserID', 'Item1', 'Item2'])
    for user_id in train_df.UserID.unique():
        user_data = train_df.query('UserID == @user_id')
        validation_row = user_data.sample(1)
        validation_row.rename(columns={'ItemID': 'Item1'}, inplace=True)
        negative_item = np.random.choice(list(set(range(1, num_items)).difference(user_data.ItemID.unique())), 1)[0]
        validation_row['Item2'] = negative_item
        validation_indices.append(validation_row.index[0])
        validation_df = validation_df.append(validation_row)
    train_df.drop(validation_indices, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    return train_df, validation_df.astype(np.int64)


def create_user_items_preferences_matrix(df: pd.DataFrame) -> np.array:
    """
    Create a numpy array (nXm) of preferences (if user liked item - value of "1" (fill_value), otherwise "0")
    where n - number_of_users, m - number of items
    :return:
    """
    fill_value = 1
    preferences_matrix = np.zeros(get_num_users_items(df=df))
    for user_id in df.UserID.unique():
        np.put(preferences_matrix[user_id], df.query('UserID == @user_id').ItemID.values, fill_value)
    return preferences_matrix.astype(np.float64)


def read_data_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the data of the train & test (2 files) and return dataframes
    """
    train_df = pd.read_csv(TRAIN_DF_PATH, index_col=False)
    test_rand_df = pd.read_csv(TEST_RAND_DF_PATH)
    test_pop_df = pd.read_csv(TEST_POP_DF_PATH)
    return train_df, test_rand_df, test_pop_df


def get_validation_and_train_matrix(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.array, pd.DataFrame]:
    """
    Extracting validation dataset from the train set, saving object for saving time while debugging
    """

    if len(os.listdir((Path.cwd() / "hdf5"))) == 0:
        train_df, validation_rand_df = get_train_random_validation_data(train_df=train_df)
        train_preferences_matrix = create_user_items_preferences_matrix(df=train_df)
        save_hdf5_to_file('train_df', train_df)
        save_hdf5_to_file('validation_rand_df', validation_rand_df)
        save_hdf5_to_file('train_preferences_matrix', train_preferences_matrix)
    else:
        train_df, train_preferences_matrix, validation_rand_df = load_hdf5_file('train_df'), load_hdf5_file(
            'train_preferences_matrix'), load_hdf5_file('validation_rand_df')
    return train_df, train_preferences_matrix, validation_rand_df

_activate_random_seed()
