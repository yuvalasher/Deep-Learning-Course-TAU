import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from tqdm import tqdm
from utils import *

RANDOM_NUM = 16
PRINT_EVERY = 1
# Early Stopping Params
MIN_IMPROVEMENT = 1e-3
PATIENT_NUM_EPOCHS = 5


class BasicNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: np.array):
        scores = self.fc(x.float())
        scores = torch.sigmoid(scores)
        return scores


def train(net, x_train, y_train, x_test, y_test, is_earlystopping=True):
    train_loss = np.zeros(NUM_EPOCHS)
    test_loss = np.zeros(NUM_EPOCHS)
    test_accuracy = np.zeros(NUM_EPOCHS)
    train_accuracy = np.zeros(NUM_EPOCHS)
    best_epoch = NUM_EPOCHS - 1

    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        y_train_pred = net(x_train)
        loss = loss_fn(input=y_train_pred.reshape(-1), target=y_train.float())
        train_loss[epoch] = loss.item()
        with torch.no_grad():
            loss.backward()
            optimizer.step()
        y_test_loss, y_test_pred = test(net, x_test, y_test)
        test_loss[epoch] = y_test_loss
        train_accuracy[epoch] = calculate_accuracy(y_true=y_train, probs=y_train_pred)
        test_accuracy[epoch] = calculate_accuracy(y_true=y_test, probs=y_test_pred)
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch}/{NUM_EPOCHS},",
                  f"Train Loss: {loss.item():.4f},",
                  f"Test Loss: {y_test_loss:.4f}",
                  f"Test Accuracy: {test_accuracy[epoch]:.2f}")

        if is_earlystopping and check_earlystopping(loss=test_loss,
                                                    epoch=epoch):  # Assigment request to use it on Test set :(
            print('EarlyStopping !!!')
            best_epoch = np.argmin(test_loss[:epoch + 1])
            break
    if best_epoch == NUM_EPOCHS - 1:  # earlystopping NOT activated
        best_epoch = np.argmin(test_loss)
    train_loss = train_loss[:best_epoch + 1]
    test_loss = test_loss[:best_epoch + 1]
    train_accuracy = train_accuracy[:best_epoch + 1]
    test_accuracy = test_accuracy[:best_epoch + 1]

    plot_values_by_epochs(train_values=train_loss, test_values=test_loss, title='Loss VS Epochs')
    plot_values_by_epochs(train_values=train_accuracy, test_values=test_accuracy, title='Accuracy VS Epochs')

    plot_roc_curve(y_test, y_test_pred)
    print(f'Test Accuracy: {calculate_accuracy(y_true=y_test, probs=y_test_pred):.2f}')
    print(
        f'Best Epoch: {best_epoch}; Best Test Loss: {test_loss[best_epoch]:.2f}; Best Test Accuracy: {test_accuracy[best_epoch]:.5f}')


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


def test(net, x_test, y_test):
    with torch.no_grad():
        net.eval()
        y_test_pred = net(x_test)
        loss = loss_fn(input=y_test_pred.reshape(-1), target=y_test.float())
        test_loss = loss.item()
    return test_loss, y_test_pred


if __name__ == '__main__':
    x, y = generate_X_y(size=500, random_num=RANDOM_NUM)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_NUM)
    lr = 0.01
    NUM_EPOCHS = 500
    input_dim = x_train.shape[1]  # Num of features
    output_dim = 1
    net = BasicNN(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction='mean')
    train(net, torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(y_test),
          is_earlystopping=True)
