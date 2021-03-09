import torch
from torch import nn
from torch.nn.modules import loss
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils import *
from consts import *
from focal_loss.focal_loss import FocalLoss

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")


def save_pt_model(net: nn.Module) -> None:
    torch.save(net.state_dict(), MODEL_WEIGHTS_PATH)


def load_pt_model() -> nn.Module:
    net = AutoEncoder(input_dim=NUM_ITEMS, output_dims=OUTPUT_DIMS)
    net.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    net.eval()
    return net


class AutoEncoderCustomDataset(Dataset):
    """
    Define x, y for data of AutoEncoder
    The y is x (for reconstruction error)
    """

    def __init__(self, features: np.array):
        self.features = torch.Tensor(features).float()
        self.labels = torch.Tensor(features).float()

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dims: List[int]):
        super(AutoEncoder, self).__init__()
        encoder_layers = [nn.Dropout(0.5), nn.Linear(input_dim, output_dims[0])]
        decoder_layers = []
        for i in range(1, len(output_dims)):
            encoder_layers.append(nn.ReLU())
            encoder_layers.extend([nn.Linear(output_dims[i - 1], output_dims[i]), nn.Sigmoid()])
        self.encoder = nn.Sequential(*encoder_layers)

        for i in range(1, len(output_dims)):
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(output_dims[-i], output_dims[-i - 1]))
        decoder_layers.extend([nn.ReLU(), nn.Linear(output_dims[0], input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction


def focal_loss(input, target, alpha=ALPHA_TENSOR, gamma: float = GAMMA) -> Tensor:
    """
    For imbalanced classification - for the "survived" class - we want to punish probability which far away than
    the true label (0.2 for class "1" is far) - how much the example is "hard"
    FL(pt) = -alpha(t) * (1-pt) ^ gamma * log(pt)
    alpha(t) - function of the frequency of number of the "survived" class items
    gamma - factor of how much you want to consider the "easy" examples against the "hard" examples
    """
    pt = torch.where(target == 1, input, 1 - input)
    pt_op = torch.where(target == 1, 1 - input, input)
    ce = - torch.log(pt)
    at = torch.where(target == 1, alpha, 1 - alpha)
    focal_loss = at * (pt_op ** gamma) * ce
    return focal_loss.mean()


def train(net: nn.Module, optimizer: torch.optim, train_dataloader: DataLoader = None,
          val_dataloader: DataLoader = None, infer_df: np.array = None, is_earlystopping: bool = False) -> nn.Module:
    """
    Training loop iterating on the train dataloader and updating the model's weights.
    Inferring the validation dataloader & test dataloader, if given, to babysit the learning
    Activating cuda device if available.
    :return: Trained model
    """
    NUMBER_OF_PREDS: int = len(train_dataloader.dataset) * NUM_USERS
    train_losses: np.array = np.zeros(NUM_EPOCHS)
    train_accuracy: np.array = np.zeros(NUM_EPOCHS)
    val_losses: np.array = np.zeros(NUM_EPOCHS)
    val_accuracy: np.array = np.zeros(NUM_EPOCHS)
    train_auc: np.array = np.zeros(NUM_EPOCHS)
    train_positive_pred: int = 0
    train_positive_number: int = 0
    val_auc: np.array = np.zeros(NUM_EPOCHS)
    best_epoch: int = NUM_EPOCHS - 1

    if val_dataloader:
        untrained_val_loss, untrained_val_accuracy = infer(net, val_dataloader, loss_fn, infer_df)
        print(f'Validation Loss before training: {untrained_val_loss:.5f}')

    for epoch in range(NUM_EPOCHS):
        print(f'*************** Epoch {epoch + 1} ***************')
        train_correct_counter = 0
        train_auc_accumulated = 0
        loss_running = 0

        net.train()
        for x_train, y_train in tqdm(train_dataloader):
            # weights = torch.where(y_train.flatten() == 0, torch.tensor(0.1), torch.tensor(0.9))
            # loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
            if train_on_gpu:
                net.cuda()
                x_train, y_train = x_train.cuda(), y_train.cuda()
            optimizer.zero_grad()
            y_train_pred = net(x_train)

            loss = loss_fn(y_train_pred.flatten(), y_train.flatten())
            loss_running += loss.item()
            loss.backward()
            optimizer.step()
            train_preds = np.where(y_train_pred > 0.5, 1, 0)
            train_correct_counter += (train_preds == np.array(y_train)).sum()
            train_positive_number += get_number_of_positves(y=y_train)
            train_positive_pred += get_number_of_tp(y_true=y_train, y_pred=train_preds)

        train_losses[epoch] = loss_running / len(train_dataloader)
        train_accuracy[epoch] = train_correct_counter.item() / NUMBER_OF_PREDS
        train_auc[epoch] = train_auc_accumulated / len(train_dataloader)
        train_recall = train_positive_pred / train_positive_number * 100

        if val_dataloader:
            val_loss, val_acc = infer(net, val_dataloader, loss_fn, infer_df)
            val_losses[epoch] = val_loss
            val_accuracy[epoch] = val_acc
            # val_auc[epoch] = val_auc_val

        if is_earlystopping and val_dataloader and check_earlystopping(loss=val_losses, epoch=epoch):
            print('EarlyStopping !!!')
            best_epoch = np.argmin(val_losses[:epoch + 1])
            break
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch + 1}/{NUM_EPOCHS},",
                  f"Train loss: {train_losses[epoch]:.5f}, Train Num Correct: {train_correct_counter} "
                  f"/ {NUMBER_OF_PREDS}, Train Accuracy: {train_accuracy[epoch]:.3f}, Train Recall: {train_recall:.3f}")
            # f"Train AUC: {train_auc[epoch]:.5f}")

            if val_dataloader:
                print(f"Validation loss: {val_losses[epoch]:.5f}, Validation Accuracy: {val_accuracy[epoch]:.3f}",
                      f"Validation AUC: {val_auc[epoch]:.5f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_pt_model(net=net)

    if best_epoch != NUM_EPOCHS - 1:  # Earlystopping NOT activated
        train_losses = train_losses[:best_epoch + 1]
        val_losses = val_losses[:best_epoch + 1]
    else:
        best_epoch = np.argmin(val_losses)

    print(
        f'Best Epoch: {best_epoch + 1}; Best Validation Loss: {val_losses[best_epoch]:.4f}')
    if val_dataloader:
        print('val_accuracy', val_accuracy)
        print('val_loss', val_loss)
    print(train_losses)
    plot_values_by_epochs(train_values=train_losses, test_values=val_losses)
    return net


def infer(net: nn.Module, infer_dataloader: DataLoader, loss_fn: loss, infer_df: np.array, test_df: pd.DataFrame = None,
          export_path: Path = None) -> Tuple[float, float]:
    """
    Run the model on x_infer (both validation and test) and calculate the loss of the predictions.
    The model run on evaluation mode and without updating the computational graph (no_grad)
    In the AutoEncoder task - taking the x_train for inference, reconstructing the user vector
    and checking which of the 2 items for a given user (in validation, test) is higher and classify it
    as "1", as it more likely for the user to like that item.
    :param infer_mode: 'test' for export results to csv & not checking accuracy
    :return loss, accuracy
    """
    net.eval()
    running_loss = 0
    infer_correct_counter = 0
    infer_auc_accumulated = 0

    for x, y in infer_dataloader:
        # weights = torch.where(y.flatten() == 0, torch.tensor(0.1), torch.tensor(0.9))
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
        with torch.no_grad():
            if train_on_gpu:
                x, y = x.cuda(), y.cuda()
            y_pred = net(x)
            y_true = get_classification_bit_gt(infer_df=infer_df)
            infer_preds = classify_preference_for_each_user(infer_df=infer_df, y_pred=y_pred)
            infer_loss = loss_fn(y_pred.flatten(), y.flatten())
            infer_correct_counter += (y_true == infer_preds).sum()

            if test_df is not None and export_path is not None:
                export_results_to_csv(test_df=test_df, bit_classification=infer_preds)

        running_loss += infer_loss
    return running_loss / len(infer_dataloader), infer_correct_counter / len(infer_dataloader.dataset)


if __name__ == '__main__':
    TRAIN: bool = True
    train_df, test_rand_df, test_pop_df = read_data_files()
    train_df, train_preferences_matrix, validation_rand_df = get_validation_and_train_matrix(train_df=train_df)
    loss_fn = nn.BCELoss()
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = FocalLoss(alpha=ALPHA_TENSOR, gamma=GAMMA)
    # loss_fn = focal_loss

    if TRAIN:
        train_dataloader = DataLoader(AutoEncoderCustomDataset(train_preferences_matrix), batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(AutoEncoderCustomDataset(train_preferences_matrix), batch_size=NUM_USERS)

        ae = AutoEncoder(input_dim=NUM_ITEMS, output_dims=OUTPUT_DIMS)
        # print_trainable_params(ae)

        optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        net = train(net=ae, optimizer=optimizer, train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader, is_earlystopping=True, infer_df=validation_rand_df)
        # print('End of Training - Infer mode:')
        # infer(net, val_dataloader, loss_fn, infer_df)
        save_pt_model(net=net)
        print(1)
    else:  # Infer for test results
        net = load_pt_model()
        test_dataloader = DataLoader(AutoEncoderCustomDataset(train_preferences_matrix), batch_size=NUM_USERS)
        _, _ = infer(net=net, infer_dataloader=test_dataloader, loss_fn=loss_fn, infer_df=test_rand_df,
                     export_path=EXPORT_RANDOM_RESULTS_PATH)
        _, _ = infer(net=net, infer_dataloader=test_dataloader, loss_fn=loss_fn, infer_df=test_rand_df,
                     export_path=EXPORT_POP_RESULTS_PATH)
