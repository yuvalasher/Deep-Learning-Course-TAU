import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from tqdm import tqdm
from utils import *
from consts import PRINT_EVERY
from data_generation import generate_regression_data


class RegressionNN(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(RegressionNN, self).__init__()
        layers = [nn.Linear(input_dim, output_dims[0])]
        for i in range(1, len(output_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(output_dims[i - 1], output_dims[i]))
        self.fc_layers = nn.ModuleList(layers)

    def forward(self, x: np.array):
        for layer in self.fc_layers:
            x = layer(x)
        return x


def train(net, x_train, y_train, x_test, y_test, is_earlystopping):
    train_loss: np.array = np.zeros(NUM_EPOCHS)
    test_loss: np.array = np.zeros(NUM_EPOCHS)
    test_MSE: np.array = np.zeros(NUM_EPOCHS)
    train_MSE: np.array = np.zeros(NUM_EPOCHS)
    best_epoch: int = NUM_EPOCHS - 1
    loss_test_before_train, _ = test(net=net, loss_fn=loss_fn, x_test=x_test, y_test=y_test)
    print(f'Loss before train: {loss_test_before_train}')
    for epoch in tqdm(range(NUM_EPOCHS)):
        net.train()
        optimizer.zero_grad()
        y_train_pred = net(x_train.float())
        loss = loss_fn(input=y_train_pred.reshape(-1), target=y_train.float())
        loss.backward()
        optimizer.step()
        train_loss[epoch] = loss.item()
        test_loss_epoch, y_test_pred = test(net, loss_fn, x_test.float(), y_test.float())

        test_loss[epoch] = test_loss_epoch
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch}/{NUM_EPOCHS},",
                  f"Train Loss: {loss.item():.6f},",
                  f"Test Loss: {test_loss_epoch:.6f}")

        if is_earlystopping and check_earlystopping(loss=test_loss,
                                                    epoch=epoch):  # Assigment request to use it on Test set :(
            print('EarlyStopping !!!')
            best_epoch = np.argmin(test_loss[:epoch + 1])
            break
    if best_epoch != NUM_EPOCHS - 1:  # earlystopping NOT activated
        train_loss = train_loss[:best_epoch + 1]
        test_loss = test_loss[:best_epoch + 1]
        train_MSE = train_MSE[:best_epoch + 1]
        test_MSE = test_MSE[:best_epoch + 1]
    else:
        best_epoch = np.argmin(test_loss)
    plot_values_by_epochs(train_values=train_loss, test_values=test_loss, title='Loss VS Epochs')
    # plot_values_by_epochs(train_values=train_MSE, test_values=test_MSE, title='MSE VS Epochs')
    print(f'Best Epoch: {best_epoch}; Best Test Loss: {test_loss[best_epoch]:.2f}; Best Test MSE: {test_MSE[best_epoch]:.5f}')


if __name__ == '__main__':
    x, y = generate_regression_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    lr = 0.001
    NUM_EPOCHS = 4000
    input_dim = x_train.shape[1]  # Num of features
    # output_dims = [32, 64, 128, 256, 128, 64, 32, 1]
    output_dims = [10,20,50,30,10,1]
    # output_dims = [1]
    net = RegressionNN(input_dim=input_dim, output_dims=output_dims)
    optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    train(net=net, x_train=torch.tensor(x_train), y_train=torch.tensor(y_train), x_test=torch.tensor(x_test),
          y_test=torch.tensor(y_test), is_earlystopping=True)
