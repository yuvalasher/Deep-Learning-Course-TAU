import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from tqdm import tqdm
from utils import *
from data_generation import generate_X_y

RANDOM_NUM = 16
PRINT_EVERY = 1

class BasicNNSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicNNSoftmax, self).__init__()
        print('input_dim', input_dim, 'output_dim', output_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: np.array):
        scores = self.fc(x.float())
        # scores = torch.sigmoid(scores)
        return scores


def train(net, x_train, y_train, x_test, y_test):
    train_loss = np.zeros(NUM_EPOCHS)
    test_loss = np.zeros(NUM_EPOCHS)
    test_accuracy = np.zeros(NUM_EPOCHS)
    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        scores = net(x_train)
        print(scores.shape, y_train.shape)
        loss = loss_fn(input=scores, target=y_train)
        train_loss[epoch] = loss.item()
        with torch.no_grad():
            loss.backward()
            optimizer.step()
        y_test_loss, y_test_pred = test(net, x_test, y_test)
        test_loss[epoch] = y_test_loss
        test_accuracy[epoch] = calculate_accuracy(y_true=y_test, probs=y_test_pred, softmax=True)
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch}/{NUM_EPOCHS},",
                  f"Train Loss: {loss.item():.4f},",
                  f"Test Loss: {y_test_loss:.4f}",
                  f"Test Accuracy: {test_accuracy[epoch]:.2f}")
    plot_values_by_epochs(train_values=train_loss, test_values=test_loss, title='Loss vs Epochs')
    # plot_roc_curve(y_test, y_test_pred)


def test(net, x_test, y_test):
    with torch.no_grad():
        net.eval()
        y_test_pred = net(x_test)
        loss = loss_fn(input=y_test_pred, target=y_test)
        test_loss = loss.item()
    return test_loss, y_test_pred


if __name__ == '__main__':
    x, y = generate_X_y(size=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    lr = 0.01
    NUM_EPOCHS = 500
    input_dim = x_train.shape[1]  # Num of features
    output_dim = 2
    net = BasicNNSoftmax(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train(net, torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(y_test))
