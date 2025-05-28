import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.utils import accuracy_fn
import matplotlib.pyplot as plt

## MS2


class MLP(nn.Module):
    """
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input
        return self.model(x)

    
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_layers = [256,128,64], dropout_p = 0.3):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the inputTrain set: accuracy = 68.183% - F1-score = 0.147641
Validation set:  accuracy = 68.030% - F1-score = 0.125328
            n_classes (int): number of classes to predict
        """
        super().__init__()

        layers = []

        in_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))             # Dropout for regularisation
            in_size = h

        layers.append(nn.Linear(in_size, n_classes))
        self.model = nn.Sequential(*layers)

        # Initialise weights
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
#        x = x.view(x.size(0), -1)  # flatten input
        return self.model(x)

class CNN(nn.Module):
    """ CNN expects inputs of shape (N, 3, 28, 28). """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()

        # First convolutional layer: 3 input channels -> 6 output channels
        self.conv2d1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, padding=2)
        
        # Second convolutional layer: 6 -> 16 channels
        self.conv2d2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        
        # After conv/pooling we flatten for fully-connected layers
        # From the output size after pooling: 28 → 14 → 7, channels: 16
        # So final feature map size: 16 × 7 × 7 = 784
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # Apply first convolution + ReLU + Max Pooling
        x = F.max_pool2d(F.relu(self.conv2d1(x)), kernel_size=2)

        # Apply second convolution + ReLU + Max Pooling
        x = F.max_pool2d(F.relu(self.conv2d2(x)), kernel_size=2)

        # Flatten for FC layers
        x = torch.flatten(x, 1)  # or x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Final output layer (no softmax, returns logits)
        x = self.fc3(x)
        return x


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer =  torch.optim.SGD(model.parameters(), lr=lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        # epoch_losses = []  # List to store loss for each epoch
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)
            

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        # Training.
        self.model.train()
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch

            # 5.2 Run forward pass --> Sequentially call the layers in order
            logits = self.model(x)
            
            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(logits, y)
            
            # 5.4 Run backward pass --> Compute gradient of the loss directions.
            loss.backward()
            
            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()
            
            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad()

            print('\r[Epoch {}/{}] Batch {}/{} - Loss: {:.4f}'.format(
            ep + 1, self.epochs, it + 1, len(dataloader), loss.item(),
            end=''))

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
                # Validation.
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]  # we don't need labels during prediction
                logits = self.model(x)
                preds = logits.argmax(dim=1)  # shape: (batch_size,)
                all_preds.append(preds)

        # Concatenate all batches
        pred_labels = torch.cat(all_preds, dim=0)  # shape: (N,)
        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
