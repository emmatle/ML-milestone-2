import numpy as np
from torchinfo import summary
from src.utils import accuracy_fn, get_n_classes, macrof1_fn
from src.methods.deep_network import CNN, MLP, Trainer
from src.data import load_data
import torch
import time
import io
import sys
import re
import itertools
from types import SimpleNamespace
from sklearn.model_selection import ParameterGrid
from medmnist import DermaMNIST
from torch.utils.data import TensorDataset, DataLoader
import argparse

# Assuming MLP, CNN, Trainer, load_data, accuracy_fn, macrof1_fn, get_n_classes are defined in the previous cells

def main(args=None):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
                          If None, argparse will parse sys.argv.
    """
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
    parser.add_argument('--optim', type=str, default="adamw", help="Optimizer to use for training, can be 'sgd', 'adam', 'adamw'" )
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout p")
    parser.add_argument('--decay', type=float, default=1e-4, help="weight decay for adam")
    parser.add_argument('--kernel', type=int, default=3, help="kernel size")
    parser.add_argument('--padding', type=int, default=1, help="padding size")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # "args_parsed" will keep in memory the arguments and their values,
    # which can be accessed as "args_parsed.data", for example.
    # Pass arguments directly to parse_args if provided, otherwise parse sys.argv
    if args is None:
        args_parsed = parser.parse_args()
    else:
        # Convert SimpleNamespace to a list of strings that argparse can parse
        # This is a simplified conversion; a more robust solution might involve
        # manually setting attributes on the parser namespace.
        arg_list = []
        for key, value in vars(args).items():
            # Only include hidden_layers if the model type is mlp
            if key == 'hidden_layers' and args.nn_type != 'mlp':
                continue

            if isinstance(value, list):
                arg_list.append(f'--{key}')
                arg_list.extend([str(v) for v in value])
            elif isinstance(value, bool):
                 if value:
                     arg_list.append(f'--{key}')
            else:
                arg_list.append(f'--{key}')
                arg_list.append(str(value))

        args_parsed = parser.parse_args(arg_list)


    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data()

    ##for CNN method i can't flatten my data
    if args_parsed.nn_type == "mlp":
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Compute mean and standard deviation based on the dimensionality of xtrain
    if args_parsed.nn_type == "mlp":
    # Flattened data (2D)
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)
        stds[stds==0]=1
    if args_parsed.nn_type == "cnn":
    # Original data (3D or higher)
        means = np.mean(xtrain, axis=(0, 1, 2))
        stds = np.std(xtrain, axis=(0, 1, 2))
        stds[stds==0]=1


    # Normalize the data
    xtrain = (xtrain - means) / stds
    xtest = (xtest - means) / stds


    # Make a validation set
    if not args_parsed.test:
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

    if args_parsed.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available")
        else:
            device = torch.device("cuda")

    elif args_parsed.device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
             device = torch.device("mps")
        else:
            print("MPS not available")


    # Neural Networks (MS2)

    # 3. Initialize the method you want to use
    n_classes = get_n_classes(ytrain)


    if args_parsed.nn_type == "dummy":
        # Assuming DummyClassifier is defined elsewhere if needed
        # model = DummyClassifier(arg1=1, arg2=2)
        # preds_train = model.fit(xtrain, ytrain)
        # preds = model.predict(xtest)
        print("DummyClassifier not implemented in this code snippet.")
        return

    elif args_parsed.nn_type == "mlp":
        input_size = xtrain.shape[1]
        model = MLP(input_size, n_classes, hidden_layers=args_parsed.hidden_layers, dropout_p=args_parsed.dropout)
        summary(model)
        model = model.to(device)
        method_obj = Trainer(model, lr=args_parsed.lr, epochs=args_parsed.max_iters, batch_size=args_parsed.nn_batch_size, weight_decay=args_parsed.decay)
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)


    elif args_parsed.nn_type == "cnn":
        # Permute dimensions for CNN (N, H, W, C) -> (N, C, H, W)
        xtrain = xtrain.transpose(0,3,1,2)
        xtest = xtest.transpose(0,3,1,2)
        input_channels = xtrain.shape[1]
        model = CNN(input_channels, n_classes, kernel_size=args_parsed.kernel, padding=args_parsed.padding)
        summary(model)
        model = model.to(device)
        method_obj = Trainer(model, lr=args_parsed.lr, epochs=args_parsed.max_iters, batch_size=args_parsed.nn_batch_size, optimizer=args_parsed.optim ,weight_decay=args_parsed.decay)
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


# The hyperparameter tuning loop remains mostly the same, but now it calls `main(args=args)`
if __name__ == '__main__':
    # Shared hyperparameters across both architectures
    common_grid = {
        'nn_batch_size': [16, 32, 64],
        'lr':            [1e-2, 1e-3, 1e-4],
        'optim':         ['adam', 'adamw'],
        'dropout':       [0.2, 0.5],
        'decay':         [1e-4, 1e-5],
        'max_iters':     [30],
        'test':          [False]
    }

    # MLP-specific grid (commented out to exclude MLP experiments)
    # mlp_grid = {
    #     **common_grid,
    #     'nn_type':      ['mlp'],
    #     'hidden_layers': [[256, 128], [512, 256, 128], [256, 128, 64]],
    # }

    # CNN-specific grid
    cnn_grid = {
        **common_grid,
        'nn_type':      ['cnn'],
        'kernel':        [3, 5],
        'padding':       [1, 2],
    }

    # Create experiment lists - ONLY INCLUDE CNN EXPERIMENTS
    # mlp_experiments = list(ParameterGrid(mlp_grid))
    cnn_experiments = list(ParameterGrid(cnn_grid))
    # experiments = mlp_experiments + cnn_experiments
    experiments = cnn_experiments # Only run CNN experiments
    print(f"Running {len(experiments)} experiments (0 MLP + {len(cnn_experiments)} CNN)...\n")

    # Result tracking per architecture - ONLY TRACK CNN RESULTS
    best_results = {
        # 'mlp': {'accuracy': -1.0, 'f1': -1.0, 'params': None, 'output': ''},
        'cnn': {'accuracy': -1.0, 'f1': -1.0, 'params': None, 'output': ''}
    }

    # Regex to parse validation metrics
    val_metrics_pattern = re.compile(
        r"Validation set:\s+accuracy\s*=\s*(\d+(?:\.\d+)?)%\s*-\s*F1-score\s*=\s*(\d+(?:\.\d+)?)"
    )

    for i, params in enumerate(experiments):
        print(f"\n=== Experiment {i+1}/{len(experiments)} ===")
        print("Params:", params)

        # Auto-detect device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        args_namespace = SimpleNamespace(
            nn_type=params['nn_type'],
            nn_batch_size=params['nn_batch_size'],
            lr=params['lr'],
            optim=params['optim'],
            dropout=params['dropout'],
            decay=params['decay'],
            max_iters=params['max_iters'],
            test=params['test'],
            device=device,
            # Add CNN specific params
            kernel=params['kernel'],
            padding=params['padding'],
            # Include hidden_layers with a default empty list for MLP, but it won't be used
            hidden_layers=[]
        )


        # Capture stdout
        buffer = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer
        try:
            # Pass the namespace directly, main will handle constructing arg_list
            main(args=args_namespace)
        except Exception as e:
            sys.stdout = sys_stdout
            print(f"Experiment {i+1} failed with error: {e}")
            print("Captured Output:\n", buffer.getvalue())
            continue
        finally:
            sys.stdout = sys_stdout

        output = buffer.getvalue()
        print(output)

        # Extract and record metrics
        match = val_metrics_pattern.search(output)
        if match:
            val_acc = float(match.group(1))
            val_f1 = float(match.group(2))
            current_best = best_results[params['nn_type']]
            if val_acc > current_best['accuracy'] or (val_acc == current_best['accuracy'] and val_f1 > current_best['f1']):
                best_results[params['nn_type']] = {
                    'accuracy': val_acc,
                    'f1': val_f1,
                    'params': params,
                    'output': output
                }

    # Print best for each
    for arch in ['cnn']: # Only print for CNN
        print(f"\n=== Best Result for {arch.upper()} ===")
        if best_results[arch]['params']:
            print(f"Accuracy: {best_results[arch]['accuracy']:.3f}%")
            print(f"F1-score: {best_results[arch]['f1']:.6f}")
            print("Params:", best_results[arch]['params'])
            print("Output:\n", best_results[arch]['output'])
        else:
            print("No successful runs.")