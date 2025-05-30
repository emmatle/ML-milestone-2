import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer, ResNet18
from src.methods.dummy_methods import DummyClassifier
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import matplotlib as plt
import torch
import time

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data()

    ##for CNN method i can't flatten my data
    if args.nn_type == "mlp":
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Compute mean and standard deviation based on the dimensionality of xtrain
    if args.nn_type == "mlp":
    # Flattened data (2D)
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)
        stds[stds==0]=1
    if args.nn_type == "cnn":
    # Original data (3D or higher)
        means = np.mean(xtrain, axis=(0, 1, 2))
        stds = np.std(xtrain, axis=(0, 1, 2))
        stds[stds==0]=1

    # Normalize the data
    xtrain = (xtrain - means) / stds
    xtest = (xtest - means) / stds


    # Make a validation set
    if not args.test:
    # Create a validation set: we will use 15% of the data as validation. As we will split the data using the first and last
    # segments of the set, we will first shuffle it to avoid bias impacting validation.
        n_train = len(xtrain)

        indices = np.random.permutation(n_train)  # Get shuffled indices
        xtrain, ytrain = xtrain[indices], ytrain[indices]  # Apply the same shuffle to both arrays

        n_val = int(np.floor(0.15 * n_train))  # Number of validation samples
        xtest, ytest = xtrain[-n_val:], ytrain[-n_val:]  # Last 15% of the training set as validation
        xtrain, ytrain = xtrain[:-n_val], ytrain[:-n_val]  # Keep the first 85% as training 
        pass

    ## 3. Initialize the method you want to use.
    ## 3. Add GPU support

    device = torch.device("cpu") # Default

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available")
        else: 
            device = torch.device("cuda")

    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available")
        else:
            device = torch.device("mps")

    # Neural Networks (MS2)

    # 3. Initialize the method you want to use
    n_classes = get_n_classes(ytrain)


    if args.nn_type == "dummy":
        model = DummyClassifier(arg1=1, arg2=2)
        preds_train = model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

    elif args.nn_type == "mlp":
        input_size = xtrain.shape[1]
        model = MLP(input_size, n_classes, hidden_layers=args.hidden_layers, dropout_p=args.dropout)
        summary(model)
        model = model.to(device)
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, weight_decay=args.decay)
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)

    elif args.nn_type == "cnn":
        xtrain = xtrain.transpose(0,3,1,2)
        xtest = xtest.transpose(0,3,1,2)
        input_channels = xtrain.shape[1]

        if args.resnet:
            model = ResNet18(n_classes)
        else:
            model = CNN(input_channels, n_classes, kernel_size=args.kernel, padding=args.padding)
        summary(model)
        model = model.to(device)
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, optimizer=args.optim ,weight_decay=args.decay)
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)

    # 4. Evaluate

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")

    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[256,128])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for methods with learning rate")
    parser.add_argument('--optim', type=str, default="adamw", help="Optimizer to use for training, can be 'sgd', 'adam', 'adamw'" )
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout p")
    parser.add_argument('--decay', type=float, default=1e-4, help="weight decay for adam")
    parser.add_argument('--kernel', type=int, default=5, help="kernel size")
    parser.add_argument('--padding', type=int, default=2, help="padding size")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--resnet', action="store_true", help="Use the Resnet18 model to classify")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example. 256
    args = parser.parse_args()
    main(args)