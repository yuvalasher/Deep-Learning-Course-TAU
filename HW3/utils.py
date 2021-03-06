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


def calcualte_recall(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculating the accuracy of the "1" labels (recall) - querying the indices of the positive values in y_true and check the value
    of the prediction in y_pred
    :return: accuracy of the positive label
    """
    positive_indices = np.where(y_true.flatten() == 1)[0]
    return (y_pred.flatten()[positive_indices].sum() / len(positive_indices)).sum()


def get_number_of_tp(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculating the number of positive examples which are true positive
    :return: accuracy of the positive label
    """
    positive_indices = np.where(y_true.flatten() == 1)[0]
    return y_pred.flatten()[positive_indices].sum()


def get_number_of_positves(y: np.array) -> int:
    """
    Calculate the number of positives in an array
    :param y:
    :return:
    """
    return len(np.where(y.flatten() == 1)[0])


def get_mask_matrix_by_infer_df(infer_df: np.array, y_pred_shape: Tuple[int, int]) -> np.array:
    """
    Create mask matrix of each user to the infer items in the infer_df
    The mask contains "1" & "0" for each item for each user, if the item belongs in the dataframe,
    we put "1" in the corresponds location in the matrix, otherwise, "0".
    For each user there are 2 items concatenated to users_items.
    :return: mask array (ma)
    """
    users_items = [[item_1 - 1, item_2 - 1] for item_1, item_2 in zip(infer_df[:, 1], infer_df[:, 2])]
    ma = np.zeros(y_pred_shape)
    for user_id, user_items in zip(infer_df[:, 0], users_items):
        np.put(ma[user_id - 1], user_items, 1)
    return ma


def get_user_items_scores_by_mask_matrix(y_pred_scores: np.array, mask_matrix: np.array) -> np.array:
    """
    From y_pred_scores & mask_matrix, we create (NUM_USERS, 2) array for 2 items
    for each user with their value
    ma = mask array
    :return: array with the items belong to user
    """
    ma = y_pred_scores.detach().numpy() * mask_matrix
    return ma[ma.nonzero()].reshape(-1, 2)


def classify_preference_for_each_user(infer_df: np.array, y_pred: np.array) -> np.array:
    """
    Classify the most likely each user to like the most from 2 items (by infer dataframe) - for each user, classify the
    item is more likely as the location (scores of 0.5, 0.9 -> results to "1" as the second location has better reconstruction)
    np.argmax((df*ma)[(df * ma).nonzero()].reshape(-1,2), axis=1)
    The probabilities in user_items_scores arrange from the the lower value of the items to the highest, so we need to flip
    the classification of the bit if in infer_df there arrangement is different (example in get_flip_bit_for_infer_df)
    :return: Index of the most likely item for each user
    """
    mask_matrix = get_mask_matrix_by_infer_df(infer_df=infer_df, y_pred_shape=y_pred.shape)
    user_items_scores = get_user_items_scores_by_mask_matrix(mask_matrix=mask_matrix, y_pred_scores=y_pred)
    y_pred = np.argmax(user_items_scores, axis=1)
    flip_bit = get_flip_bit_for_infer_df(infer_df=infer_df)
    flipped_y_pred = np.where(flip_bit == 0, y_pred, 1 - y_pred)
    return flipped_y_pred


def get_flip_bit_for_infer_df(infer_df: np.array):
    """
    As the mask in classify_preference_for_each_user make the probabilities order ascending (the first probability belongs
    to the smaller item), we need to custom the flip_bit to indicate that the results the np.argmax(user_items_scores)
    should be flipped. E.X: user: 111, Item1: 1512, Item2: 1217, the probabilities will be arrange as np.argmax{P(1217), P(1512)},
    nevertheless, it should be the opposite, so the flip_bit should be "1" and flip the bit classification
    :return: flip bit for each row in the infer df
    """
    return (infer_df[:, 1] == np.max(infer_df[:, 1:3], axis=1)).astype(int)


def export_results_to_csv(test_df: np.array, bit_classification: np.array, export_path: Path) -> None:
    """
    Export bit classification for test data in format of <userId>, <item1>, <item2>, <bitClassifiction>
    """
    test_df = pd.DataFrame(test_df).rename(columns={0: 'UserID', 1: 'Item1', 2: 'Item2'})
    test_df['bit'] = bit_classification
    test_df.to_csv(export_path, index=False)


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


def save_hdf5_to_file(obj_name: str, obj: Any) -> None:
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


def get_num_users_items(df) -> Tuple[int, int]:
    if type(df) == pd.DataFrame:
        df = df.to_numpy()
    return len(np.unique(df[:, 0])), len(np.unique(df[:, 1])) + 1


def get_train_random_validation_data(train_df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Creating random validation set from the train set - for each user, there is only one record in the validation
    The item the user liked is in the first column ("Item1")
    """
    train_df = train_df.to_numpy()
    num_users, num_items = get_num_users_items(df=train_df)
    validation_indices = []
    validation_df = np.zeros((num_users, 3))
    for user_id in np.unique(train_df[:, 0]):
        user_items = np.where(train_df[:, 0] == user_id)[0]
        validation_user_item_index = np.random.choice(user_items)
        validation_user_item = train_df[validation_user_item_index][1]
        negative_user_item = np.random.choice(list(set(range(1, num_items)).difference(user_items)), 1)[0]

        validation_df[user_id - 1][0] = user_id
        validation_df[user_id - 1][1] = validation_user_item
        validation_df[user_id - 1][2] = negative_user_item
        validation_indices.append(validation_user_item_index)

    train_df = np.delete(train_df, validation_indices, axis=0)
    return train_df, validation_df.astype(np.int64)


def create_user_items_preferences_matrix(df: pd.DataFrame) -> np.array:
    """
    Create a numpy array (nXm) of preferences (if user liked item - value of "1" (fill_value), otherwise "0")
    where n - number_of_users, m - number of items
    :return:
    """
    fill_value = 1
    preferences_matrix = np.zeros(get_num_users_items(df=df))
    for user_id in np.unique(df[:, 0]):
        np.put(preferences_matrix[user_id - 1], df[np.where(df[:, 0] == user_id)][:, 1] - 1, fill_value)
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
