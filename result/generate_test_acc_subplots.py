import matplotlib
matplotlib.use('Agg')

import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from seed_config import dataset_info

def main():
    parser = argparse.ArgumentParser(description='Generate test accuracy plot with subplots for multiple datasets.')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir


    frameworks = ['FedAPM', 'FedAvg', 'FedAlt', 'FedSim', 'FedProx']

    # Prepare the figure with subplots
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 4), sharey=True)
    if num_datasets == 1:
        axes = [axes]  # Make it iterable

    for ax, dataset_name in zip(axes, datasets):
        if dataset_name not in dataset_info:
            print(f"Dataset {dataset_name} not recognized.")
            continue

        model_name = dataset_info[dataset_name]['model']
        seeds = dataset_info[dataset_name]['seeds']

        # Regular expression to parse filenames
        filename_pattern = re.compile(rf"^{dataset_name}_{model_name}_(.*?)_random_seed_(\d+)_users_(\d+)_.*\.json$")

        # Build mapping from (framework, seed) to file paths
        file_mapping = defaultdict(list)
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.json'):
                    match = filename_pattern.match(filename)
                    if match:
                        framework, seedvalue, userval = match.groups()
                        seedvalue = int(seedvalue)
                        if framework not in frameworks:
                            continue
                        if seedvalue not in seeds:
                            continue
                        filepath = os.path.join(root, filename)
                        key = (framework, seedvalue)
                        file_mapping[key].append(filepath)

        # Initialize data structures to collect test_acc over iterations
        acc_data = defaultdict(list)  # framework -> list of test_acc arrays

        # Iterate over files and collect test_acc
        for (framework, seed), filepaths in file_mapping.items():
            for filepath in filepaths:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    test_acc = data.get('test_acc', [])
                    if test_acc:
                        acc_data[framework].append(test_acc)

        # Plot data for each framework
        for framework in frameworks:
            if framework in acc_data and acc_data[framework]:
                # Pad shorter arrays with last value to match lengths
                max_length = max(len(acc) for acc in acc_data[framework])
                acc_arrays = []
                for acc in acc_data[framework]:
                    if len(acc) < max_length:
                        acc.extend([acc[-1]] * (max_length - len(acc)))
                    acc_arrays.append(acc)
                acc_arrays = np.array(acc_arrays)
                avg_acc = np.mean(acc_arrays, axis=0)
                ax.plot(avg_acc, label=framework)
        if dataset_name == 'cifar10':
            dataset_name = 'CIFAR10'
        elif dataset_name == 'crema_d':
            dataset_name = 'CREMA-D'
        elif dataset_name == 'crisis_mmd':
            dataset_name = 'CrisisMMD'
        elif dataset_name == 'ku_har':
            dataset_name = 'KU-HAR'
        ax.set_title(dataset_name)
        ax.set_xlabel('Iteration')
        if ax == axes[0]:
            ax.set_ylabel('Test Accuracy')
        ax.legend()

    plt.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    # Save the plot as PNG
    plot_file_png = os.path.join(result_dir, 'test_acc_subplots.png')
    plt.savefig(plot_file_png)

    # Save the plot as PDF
    plot_file_pdf = os.path.join(result_dir, 'test_acc_subplots.pdf')
    plt.savefig(plot_file_pdf)

    plt.close()

    print(f"Grouped training acc saved at {plot_file_png} and {plot_file_pdf}")
    
    

if __name__ == "__main__":
    main()
