import torch
from torch import nn
from torch.nn.modules import loss
from torch import Tensor
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils import *
from consts import *

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")


def connect_net_to_fc(net: nn.Module, num_features: int, num_classes: int = NUM_CLASSES):
    """
    Defining the last fully connected layer to the given net using nn.Linear
    :return: Concatenated network
    """
    net.fc = nn.Linear(num_features, num_classes)
    if train_on_gpu:
        net.cuda()
    return net


class CIFARCustomDataset(Dataset):
    """
    The dataset object used to read the data
    """

    def __init__(self, features: np.array, labels: np.array):
        assert features.shape[0] == labels.shape[0]
        self.features = features.float()
        self.labels = torch.Tensor(labels).long()

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]


def get_dataset_by_classes_subset(dataset: Dataset, num_samples: int, classes=CLASSES) -> Dataset:
    """
    Get the relevant classes' data in the dataset by class names
    """
    dog_indices, cat_indices = [], []
    dog_idx, cat_idx = dataset.class_to_idx[classes[0]], dataset.class_to_idx[classes[1]]

    for idx in range(len(dataset)):
        class_idx = dataset[idx][1]
        if class_idx == dog_idx:
            dog_indices.append(idx)
        elif class_idx == cat_idx:
            cat_indices.append(idx)

    dog_indices = np.array(dog_indices)[np.random.choice(range(len(dog_indices)), size=num_samples, replace=False)]
    cat_indices = np.array(cat_indices)[np.random.choice(range(len(cat_indices)), size=num_samples, replace=False)]
    new_dataset = Subset(dataset, list(np.concatenate([dog_indices, cat_indices])))
    return new_dataset


def save_pt_model(net: nn.Module) -> None:
    torch.save(net.state_dict(), 'Models/weights.pt')


def train(net: nn.Module, optimizer: torch.optim, train_dataloader: DataLoader = None,
          val_dataloader: DataLoader = None, is_earlystopping: bool = False) -> nn.Module:
    """
    Training loop iterating on the train dataloader and updating the model's weights.
    Inferring the validation dataloader & test dataloader, if given, to babysit the learning
    Activating cuda device if available.
    :return: Trained model
    """
    train_losses: np.array = np.zeros(NUM_EPOCHS)
    train_accuracy: np.array = np.zeros(NUM_EPOCHS)
    val_losses: np.array = np.zeros(NUM_EPOCHS)
    val_accuracy: np.array = np.zeros(NUM_EPOCHS)
    train_auc: np.array = np.zeros(NUM_EPOCHS)
    val_auc: np.array = np.zeros(NUM_EPOCHS)
    best_epoch: int = NUM_EPOCHS - 1

    if val_dataloader:
        untrained_test_loss, untrained_test_accuracy, untrained_test_auc = infer(net, val_dataloader, loss_fn)
        print(f'Test Loss before training: {untrained_test_loss:.5f}')

    for epoch in range(NUM_EPOCHS):
        print(f'*************** Epoch {epoch + 1} ***************')
        train_correct_counter = 0
        train_auc_accumulated = 0
        loss_running = 0
        net.train()
        for x_train, y_train in tqdm(train_dataloader):
            if x_train.shape[-1] == 224:
                y_train = torch.tensor(np.where(y_train == 3, 0, 1)).long()
            if train_on_gpu:
                net.cuda()
                x_train, y_train = x_train.cuda(), y_train.cuda()
            optimizer.zero_grad()
            y_train_pred = net(x_train)

            loss = loss_fn(y_train_pred, y_train)
            loss_running += loss.item()
            loss.backward()
            optimizer.step()
            _, train_preds = torch.max(y_train_pred, dim=1)
            train_correct_counter += torch.sum(train_preds == y_train)
            train_auc_accumulated += calculate_auc_score(y_true=y_train, y_pred=train_preds)

        train_losses[epoch] = loss_running / len(train_dataloader)
        train_accuracy[epoch] = train_correct_counter.item() / len(train_dataloader.dataset)
        train_auc[epoch] = train_auc_accumulated / len(train_dataloader)

        if val_dataloader:
            val_loss, val_acc, val_auc_val = infer(net, val_dataloader, loss_fn)
            val_losses[epoch] = val_loss
            val_accuracy[epoch] = val_acc
            val_auc[epoch] = val_auc_val

        if is_earlystopping and check_earlystopping(loss=val_losses, epoch=epoch):
            print('EarlyStopping !!!')
            best_epoch = np.argmin(val_losses[:epoch + 1])
            break

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch: {epoch + 1}/{NUM_EPOCHS},",
                  f"Train loss: {train_losses[epoch]:.5f}, Train Num Correct: {train_correct_counter} "
                  f"/ {len(train_dataloader.dataset)}, Train Accuracy: {train_accuracy[epoch]:.3f}\n",
                  f"Validation loss: {val_losses[epoch]:.5f}, Validation Accuracy: {val_accuracy[epoch]:.3f}",
                  f"Validation AUC: {val_auc[epoch]:.5f}, Train AUC: {train_auc[epoch]:.5f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_pt_model(net=net)

    if best_epoch != NUM_EPOCHS - 1:  # earlystopping NOT activated
        train_losses = train_losses[:best_epoch + 1]
        val_losses = val_losses[:best_epoch + 1]
    else:
        best_epoch = np.argmin(val_losses)

    print(
        f'Best Epoch: {best_epoch + 1}; Best Validation Loss: {val_losses[best_epoch]:.4f}')
    print('val_accuracy', val_accuracy)
    print('val_loss', val_loss)
    print(train_losses)
    plot_values_by_epochs(train_values=train_losses, test_values=val_losses)
    return net


def infer(net: nn.Module, infer_dataloader: DataLoader, loss_fn: loss) -> Tuple[float, float, float]:
    """
    Run the model on x_infer (both validation and test) and calculate the loss of the predictions.
    The model run on evaluation mode and without updating the computational graph (no_grad)
    Running on the dataloader by batches, defined in each dataset's DataLoader
    :return loss, accuracy
    """
    net.eval()
    running_loss = 0
    infer_correct_counter = 0
    infer_auc_accumulated = 0

    for x, y in infer_dataloader:
        if x.shape[-1] == 224:
            y = torch.tensor(np.where(y == 3, 0, 1)).long()
        with torch.no_grad():
            if train_on_gpu:
                x, y = x.cuda(), y.cuda()
            y_pred = net(x)
            _, preds = torch.max(y_pred, dim=1)

            infer_correct_counter += torch.sum(preds == y)
            infer_auc_accumulated += calculate_auc_score(y_true=y, y_pred=preds)

            infer_loss = loss_fn(input=y_pred, target=y)
        running_loss += infer_loss
    infer_correct_counter = infer_correct_counter.item()
    return running_loss / len(infer_dataloader), infer_correct_counter / len(infer_dataloader.dataset), infer_auc_accumulated / len(infer_dataloader)


def get_net_representation(net: nn.Module, dataloader: Dataset) -> Tuple[np.array, List[float]]:
    """
    Insert the data to the model, without fully-connected layer, which will perform the CONV layers
    to create representation of the data, without making classification
    :return: Representation of the data by the net
    """
    net.eval()

    representations_array = []
    y_array = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            print('x.shape', x.shape)
            # representations_array.append(net(x).reshape(-1))
            rep = net(x)
            print('rep.shape', rep.shape)
            representations_array.append(rep)
            y_array.append(y)
    return torch.cat(tuple(representations_array)), torch.cat(tuple(y_array))


class NN(nn.Module):
    def __init__(self, input_dim: int, output_dims: List[int]):
        super(NN, self).__init__()
        layers = [nn.Linear(input_dim, output_dims[0])]
        for i in range(1, len(output_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(output_dims[i - 1], output_dims[i]))
        self.fc_layers = nn.ModuleList(layers)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor):
        for layer in self.fc_layers:
            x = layer(x)
        # x = self.sigmoid(x)
        return x


def load_representations_data() -> Tuple[np.array, np.array, np.array, np.array]:
    train_representation_array = torch.load('train_representation_array.pt')
    y_train = torch.load('y_train.pt')
    test_representation_array = torch.load('test_representation_array.pt')
    y_test = torch.load('y_test.pt')
    y_train = np.where(y_train == 3, 0, 1)
    y_test = np.where(y_test == 3, 0, 1)
    return train_representation_array, y_train, test_representation_array, y_test


def train_and_predict_logistic_regression(X_train: np.array, y_train: np.array, X_test: np.array) -> np.array:
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    return clf.predict(X_test)


def define_net(resent_num_layers: int = RESNET_NUM_LAYERS, feature_extractor: bool = True,
               connect_head: bool = True, head: nn.Module = None) -> Tuple[nn.Module, int]:
    """
    :param resent_num_layers: define the type of the resnet
    :param feature_extractor: Indicates if the weights of the network should be freeze or not
    :return: The defined model & the number of features at the end
    """
    # TODO - how to import the clever way with importlib? importlib.import_module('torchvision.models.{}'.format(NUM))
    if resent_num_layers == 18:
        net = torchvision.models.resnet18(pretrained=True)
    else:
        net = torchvision.models.resnet34(pretrained=True)

    # Parameters of newly constructed modules have requires_grad=True by default
    for param in net.parameters():
        if feature_extractor:
            param.requires_grad = False
        else:
            param.requires_grad = True
    num_features = net.fc.in_features
    print(f'Num Features: {num_features}')

    if connect_head:
        if head is not None:
            net.fc = head
            return net, num_features
        else:
            net = connect_net_to_fc(net, num_features=num_features, num_classes=NUM_CLASSES)
    else:
        net.fc = nn.Identity()
    return net, num_features


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


def get_subset_dataset_by_classes() -> Tuple[Dataset, Dataset]:
    """
    Downloading the data of the CIFAR10, running a transform on the train and the infer data (separately), extracting
    just the relevant labels with the defined amount of samples for each class
    :return:
    """
    if os.path.exists(os.path.abspath(os.path.join(FILES_PATH, 'train_dataset.pt'))):
        print('train_dataset.pt exists')
        train_dataset, test_dataset = torch.load('train_dataset.pt'), torch.load('test_dataset.pt')
    else:
        train_dataset = datasets.CIFAR10("cifar", transform=train_transform, train=True, download=True)
        test_dataset = datasets.CIFAR10("cifar", transform=infer_transform, train=False)
        print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')
        train_dataset = get_dataset_by_classes_subset(dataset=train_dataset, num_samples=TRAIN_CLASS_NUM_SAMPLES)
        test_dataset = get_dataset_by_classes_subset(dataset=test_dataset, num_samples=TEST_CLASS_NUM_SAMPLES)
        torch.save(train_dataset, 'train_dataset.pt')
        torch.save(test_dataset, 'test_dataset.pt')
    print(f'After pruning samples - Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')
    return train_dataset, test_dataset


def create_dataset_dataloaders(train_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    """
    Wrapping Datasets into DataLoaders
    """
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'After pruning - Train samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    return train_dataloader, test_dataloader


def get_representation_data(net: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
                            save_data_on_disk: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Extracting the representation of the images both in the train and in the test data
    For each image, which transformed by train & infer transformation (defined in utils.py) - converted to the expected
    resnet representation (batch_size, 3, 224, 224), running the data in the resent pretrained net in a eval mode and
    receive for each image, vector with size of 512 which represent it.
    Load the .pt file if exists, else create and saves it
    :return:
    """
    if os.path.exists(os.path.abspath(os.path.join(FILES_PATH, 'train_representation_array.pt'))):
        train_representation_array, y_train, test_representation_array, y_test = load_representations_data()

    elif FEATURES_EXTRACTION and os.path.exists(os.path.abspath(os.path.join(FILES_PATH, 'train_dataset.pt'))):
        train_representation_array, y_train = get_net_representation(net=net, dataloader=train_dataloader)
        test_representation_array, y_test = get_net_representation(net=net, dataloader=test_dataloader)
        if save_data_on_disk:
            torch.save(train_representation_array, 'train_representation_array.pt')
            torch.save(y_train, 'y_train.pt')
            torch.save(test_representation_array, 'test_representation_array.pt')
            torch.save(y_test, 'y_test.pt')
    else:
        raise ('No representation data')
    return train_representation_array, y_train, test_representation_array, y_test


def get_representation_dataloaders(train_representation_array: Tensor, y_train: Tensor,
                                   test_representation_array: Tensor, y_test: Tensor) -> Tuple[
    DataLoader, DataLoader]:
    """
    Wrapping the Tensors of the representations from the pretrained network by a custom Dataset and inserted
     to a DataLoader with defined BATCH_SIZE from consts.py
    """
    representation_train_dataloader = DataLoader(
        CIFARCustomDataset(features=train_representation_array, labels=y_train), batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS)

    representation_test_dataloader = DataLoader(CIFARCustomDataset(features=test_representation_array, labels=y_test),
                                                batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(train_representation_array.shape, y_train.shape, test_representation_array.shape)
    return representation_train_dataloader, representation_test_dataloader


if __name__ == '__main__':
    """
    * Feature extractor: Here we remove top layers, freeze the weights of all remaining layers,
    add a new fully connected layers on top of the pretrained model.
    Then, we train only these new layers to solve the new task.
    * Fine-Tuning: Freeze the pretrained-model's weights, connect the classifier, unfreeze all the weights, 
    and re-train the entire model on the new data with very low lr
    """
    FEATURES_EXTRACTION: bool = True
    NET: bool = True
    FINE_TUNING: bool = True

    if NET:
        resent, num_features = define_net(feature_extractor=True, connect_head=True)
    else:
        resent, num_features = define_net(feature_extractor=True, connect_head=False)
    train_dataset, test_dataset = get_subset_dataset_by_classes()
    train_dataloader, test_dataloader = create_dataset_dataloaders(train_dataset, test_dataset)
    train_representation_array, y_train, test_representation_array, y_test = get_representation_data(net=resent,
                                                                                                     train_dataloader=train_dataloader,
                                                                                                     test_dataloader=test_dataloader)
    representation_train_dataloader, representation_test_dataloader = get_representation_dataloaders(
        train_representation_array, y_train, test_representation_array, y_test)

    if not NET:  # Logistic Regression
        y_test_pred = train_and_predict_logistic_regression(X_train=train_representation_array,
                                                            y_train=y_train,
                                                            X_test=test_representation_array)
        print(f'Logistic Regress Test Accuracy: {accuracy_score(y_true=y_test, y_pred=y_test_pred)}')

    else:
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCELoss()
        classifier = NN(input_dim=num_features, output_dims=[20, 10, NUM_CLASSES])

        print_trainable_params(classifier)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        classifier = train(net=classifier, optimizer=optimizer, train_dataloader=representation_train_dataloader,
                           val_dataloader=representation_test_dataloader, is_earlystopping=False)
        print('Classifier after Training')
        print(classifier)

        resent.fc = classifier  # Connecting the classifier to an End-To-End Pipeline

        if FINE_TUNING:
            print('Fine Tuning !')
            for param in resent.parameters():
                param.requires_grad = True
            # net, _ = define_net(feature_extractor=False, connect_head=False, head=classifier)
            print_trainable_params(resent)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=FINE_TUNING_LR)
            net = train(net=resent, optimizer=optimizer, train_dataloader=train_dataloader,
                        val_dataloader=test_dataloader, is_earlystopping=False)
