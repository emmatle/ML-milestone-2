import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.methods.dummy_methods import DummyClassifier
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import matplotlib as plt
import torch

'''
############## COMMENTARY SECTION ###############################################################################################
Ali: Coucou, il faudra regarder si on a le temps de faire run le truc pour savoir le run time de nos algorithms :)

#################################################################################################################################
'''


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
    #xtrain = xtrain.reshape(xtrain.shape[0], -1)
    #xtest = xtest.reshape(xtest.shape[0], -1)
    if args.nn_type != "cnn":
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

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
    """
    Tried adding this, but when run it gave an error and said that get_n_classes is not compatible with torch bc it uses numpy
    The get_n_classes fct is in scr/utils.py so i am not sure we are allowed to modify it  
    ## 3. Add GPU support
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "mps" else "cpu")
    xtrain = torch.tensor(xtrain).to(device)
    xtest = torch.tensor(xtest).to(device)
    ytrain = torch.tensor(ytrain).to(device)
    ytest = torch.tensor(ytest).to(device)
    """

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    input_size = 0 # Todo

    # 3. Initialize the method you want to use
    n_classes = get_n_classes(ytrain)
    input_size = xtrain.shape[1]

    if args.nn_type == "dummy":
        model = DummyClassifier(arg1=1, arg2=2)
        preds_train = model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

    elif args.nn_type == "mlp":
        model = MLP(input_size, n_classes, hidden_layers=args.hidden_layers)
        summary(model)
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)

    elif args.nn_type == "cnn":
        model = CNN(input_size, n_classes)
        summary(model)
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
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


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.89644


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

    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[256,128,64])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example. 256
    args = parser.parse_args()
    main(args)
