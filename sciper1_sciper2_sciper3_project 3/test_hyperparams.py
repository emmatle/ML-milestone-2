import itertools
from types import SimpleNamespace
from main import main
import torch
import io
import sys
import re

# Define the grid of parameters to test
param_grid = {
    'nn_type': ['mlp', 'cnn'],
    'nn_batch_size': [16, 32, 64],
    'hidden_layers': [[256, 128], [512, 256, 128], [256, 128, 64]],
    'lr': [1e-2, 1e-3, 1e-4],
    'optim': ['adam', 'adamw'],
    'dropout': [0.2, 0.5],
    'decay': [1e-4, 1e-5],
    'kernel': [3, 5],
    'padding': [1, 2],
    'max_iters': [30],
    'test': [False]
}

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(experiments)} hyperparameter combinations...\n")

# Result tracking per architecture
best_results = {
    'mlp': {'accuracy': -1.0, 'f1': -1.0, 'params': None, 'output': ''},
    'cnn': {'accuracy': -1.0, 'f1': -1.0, 'params': None, 'output': ''}
}

# Regex to extract metrics from validation set
val_metrics_pattern = re.compile(
    r"Validation set:\s+accuracy\s*=\s*(\d+(?:\.\d+)?)%\s*-\s*F1-score\s*=\s*(\d+(?:\.\d+)?)"
)

for i, params in enumerate(experiments):
    print(f"\n=== Experiment {i+1}/{len(experiments)} ===")
    print("Params:", params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(
        nn_type=params['nn_type'],
        nn_batch_size=params['nn_batch_size'],
        hidden_layers=params['hidden_layers'],
        lr=params['lr'],
        optim=params['optim'],
        dropout=params['dropout'],
        decay=params['decay'],
        kernel=params['kernel'],
        padding=params['padding'],
        max_iters=params['max_iters'],
        test=params['test'],
        device=device
    )

    # Capture stdout from main()
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    try:
        main(args)
    except Exception as e:
        sys.stdout = sys_stdout
        print(f"Experiment {i+1} failed with error: {e}")
        continue
    finally:
        sys.stdout = sys_stdout

    output = buffer.getvalue()
    print(output)

    # Extract accuracy and F1 from the validation set printout
    match = val_metrics_pattern.search(output)
    if match:
        val_acc = float(match.group(1))
        val_f1 = float(match.group(2))

        # Check if this is the best for the current architecture
        current_best = best_results[params['nn_type']]
        if val_acc > current_best['accuracy']:
            best_results[params['nn_type']] = {
                'accuracy': val_acc,
                'f1': val_f1,
                'params': params,
                'output': output
            }

# Print best results
for arch in ['mlp', 'cnn']:
    print(f"\n=== Best Result for {arch.upper()} ===")
    if best_results[arch]['params']:
        print(f"Accuracy: {best_results[arch]['accuracy']:.3f}%")
        print(f"F1-score: {best_results[arch]['f1']:.6f}")
        print("Params:", best_results[arch]['params'])
        print("Output:\n", best_results[arch]['output'])
    else:
        print("No successful runs.")