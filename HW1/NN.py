import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from tqdm import tqdm
from utils import *
from consts import PRINT_EVERY
from data_generation import generate_X_y


class NN(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(NN, self).__init__()
        layers = [nn.Linear(input_dim, output_dims[0])]
        for i in range(1, len(output_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(output_dims[i - 1], output_dims[i]))
        self.fc_layers = nn.ModuleList(layers)

    def forward(self, x: np.array):
        for layer in self.fc_layers:
            x = layer(x)
        scores = torch.sigmoid(x)
        return scores


def train(net, x_train, y_train, x_test, y_test, is_earlystopping):
    train_loss: np.array = np.zeros(NUM_EPOCHS)
    test_loss: np.array = np.zeros(NUM_EPOCHS)
    test_accuracy: np.array = np.zeros(NUM_EPOCHS)
    train_accuracy: np.array = np.zeros(NUM_EPOCHS)
    best_epoch: int = NUM_EPOCHS - 1
    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        y_train_pred = net(x_train.float())
        loss = loss_fn(input=y_train_pred.reshape(-1), target=y_train.float())
        train_loss[epoch] = loss.item()
        with torch.no_grad():
            loss.backward()
            optimizer.step()
        y_test_loss, y_test_pred = test(net, loss_fn, x_test.float(), y_test.float())

        test_loss[epoch] = y_test_loss
        test_accuracy[epoch] = calculate_accuracy(y_true=y_test, probs=y_test_pred)
        train_accuracy[epoch] = calculate_accuracy(y_true=y_train, probs=y_train_pred)
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch}/{NUM_EPOCHS},",
                  f"Train Loss: {loss.item():.2f},",
                  f"Test Loss: {y_test_loss:.2f}",
                  f"Train Accuracy: {train_accuracy[epoch]:.2f}",
                  f"Test Accuracy: {test_accuracy[epoch]:.4f}")

        if is_earlystopping and check_earlystopping(loss=test_loss,
                                                    epoch=epoch):  # Assigment request to use it on Test set :(
            print('EarlyStopping !!!')
            best_epoch = np.argmin(test_loss[:epoch + 1])
            break
    if best_epoch != NUM_EPOCHS - 1:  # earlystopping NOT activated
        train_loss = train_loss[:best_epoch + 1]
        test_loss = test_loss[:best_epoch + 1]
        train_accuracy = train_accuracy[:best_epoch + 1]
        test_accuracy = test_accuracy[:best_epoch + 1]
    else:
        best_epoch = np.argmin(test_loss)
    plot_values_by_epochs(train_values=train_loss, test_values=test_loss, title='Loss VS Epochs')
    plot_values_by_epochs(train_values=train_accuracy, test_values=test_accuracy, title='Accuracy VS Epochs')
    plot_roc_curve(y_test, y_test_pred)
    print(
        f'Best Epoch: {best_epoch}; Best Test Loss: {test_loss[best_epoch]:.2f}; Best Test Accuracy: {test_accuracy[best_epoch]:.5f}')


if __name__ == '__main__':
    x, y = generate_X_y(size=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    lr = 0.001
    NUM_EPOCHS = 4000
    input_dim = x_train.shape[1]  # Num of features
    output_dims = [10, 20, 30, 20, 5, 1]
    # output_dims = [1]
    net = NN(input_dim=input_dim, output_dims=output_dims)
    optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCELoss(reduction='mean')
    train(net=net, x_train=torch.tensor(x_train), y_train=torch.tensor(y_train), x_test=torch.tensor(x_test),
          y_test=torch.tensor(y_test), is_earlystopping=True)
