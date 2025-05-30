import re

def extract_best_experiment(filepath):
    best_f1 = -1.0
    best_experiment = None
    best_params = None

    with open(filepath, 'r') as file:
        content = file.read()

    # Split by experiment
    experiments = re.split(r'=== Experiment \d+/\d+ ===', content)[1:]

    for exp in experiments:
        # Extract params
        params_match = re.search(r"Params: (.*?)\n", exp)
        if not params_match:
            continue
        params = eval(params_match.group(1))

        # Extract validation F1-score
        val_f1_match = re.search(r"Validation set:.*?F1-score = ([0-9.]+)", exp)
        if val_f1_match:
            val_f1 = float(val_f1_match.group(1))
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_experiment = exp.strip()
                best_params = params

    return best_f1, best_params, best_experiment

# Usage
file_path = 'hyperparam_mlp_results.txt'
best_f1, best_params, best_experiment = extract_best_experiment(file_path)

print(f"Best Validation F1-score: {best_f1}")
print(f"Best Params: {best_params}")
print(f"\nBest Experiment Details:\n{best_experiment}")