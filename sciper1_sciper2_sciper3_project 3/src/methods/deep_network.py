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

    def __init__(self, input_size, n_classes, hidden_layers = [256,128], dropout_p = 0.2):
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
            layers.append(nn.Dropout(p=dropout_p))
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
    """ CNN expects inputs of shape (N, Ch, H, W). """

    def __init__(self, input_channels, n_classes, kernel_size=3, padding=1):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input (3 for RGB images)
            n_classes (int): number of classes to predict (7 for DermaMNIST)
            kernel_size (int) : Size of the Kernel for convolution layers (default value = 3)
            padding (int) : help preserving spatial dimensions 28x28 (default value = 1)
        """

        #We first call the constructor nn.Module to set up the PyTorch library functionalities 
        super(CNN, self).__init__()

        # We define the first convolutional layer: 3 input channels -> 6 output channels using as we said a Kernel and padding 
        self.conv2d1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=kernel_size, padding=padding)
        
        # Then the second convolutional layer: 6 channels from the previous layer -> 16 output channels still using the same kernel and padding 
        self.conv2d2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size, padding=padding)
        
        # We defined now the Fully connected layers
        # Knowing that the FC layers need a 1D input we flatten the 3D vectors produced after convolutions and pooling 
        # 16 features map each of size 7x7 that will be compressed and transformed by the FC layers 
        self.fc1 = nn.Linear(16 * 7 * 7, 120) #first FC layer (flatten input : 16 × 7 × 7 = 784, output : 120)
        self.fc2 = nn.Linear(120, 84)         #second FC layer (input : 120, output : 84)
        self.fc3 = nn.Linear(84, 32)          #third FC layer (input : 84, output : 32)
        self.fc4 = nn.Linear(32, n_classes)   #last FC layers (input : 32, output layer : 7 (one value par class on logits form)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W) (here (N, 3, 28, 28))
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # We apply the first convolution layer, ReLU and then Max Pooling with kernel size 2x2
        x = F.max_pool2d(F.relu(self.conv2d1(x)), kernel_size=2)

        # Then we apply the second convolution layer, ReLU and Adaptive Pooling 
        #This is to ensure our output has spatial dimension fixed at 7x7
        x = F.adaptive_max_pool2d(F.relu(self.conv2d2(x)), output_size=(7, 7))

        # As we said before we flatten for FC layers that expect 1D input 
        x = torch.flatten(x, 1)  # or x.view(x.size(0), -1)

        # Then we apply Fully connected layers with ReLU activations to help the model learn faster and deeper.
        x = F.relu(self.fc1(x)) #first FC layer
        x = F.relu(self.fc2(x)) #second FC layer
        x = F.relu(self.fc3(x)) #third FC layer 
        # This will be the final output layer (no softmax here)
        x = self.fc4(x)
        return x  #(returns logits)

class ResNet18(nn.Module):
    def __init__(self, num_classes=7):
        """
        Custom implementation of ResNet-18.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ResNet18, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the ResNet-18 blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        Create a ResNet block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of residual blocks.
            stride (int): Stride of the first convolutional layer in the block.

        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        A single residual block with a bottleneck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolutional layers.
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, optimizer="adamw" , weight_decay = 1e-4):
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
        self.device = next(model.parameters()).device  # Automatically infer device from model

        self.criterion = nn.CrossEntropyLoss()
        if optimizer=="adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer=="adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer=="sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

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
            
            self.scheduler.step() ##################### this is the schduler part
        
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
            x = x.to(self.device)
            y = y.to(self.device)


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

            if it % 10 == 0 or it == len(dataloader) - 1:
                print('\r[Epoch {}/{}] Batch {}/{} - Loss: {:.4f}'.format(
                    ep + 1, self.epochs, it + 1, len(dataloader), loss.item()), end='')

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
                x = x.to(self.device)
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
        train_dataset = TensorDataset(torch.from_numpy(training_data).float().to(self.device),
                                      torch.from_numpy(training_labels).to(self.device))
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
        test_dataset = TensorDataset(torch.from_numpy(test_data).float().to(self.device))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
