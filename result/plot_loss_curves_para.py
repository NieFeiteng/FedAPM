import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from seed_config import dataset_info
import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves for a given framework and varying parameter.')
    parser.add_argument('--framework', type=str, required=True, choices=['FedAPM', 'FedAvg', 'FedAlt', 'FedSim', 'FedProx'], help='Framework to plot')
    parser.add_argument('--param', type=str, required=True, choices=['lr', 'rho', 'local-ep', 'frac'], help='Parameter to vary')
    parser.add_argument('--data_dir', type=str, default='../save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='./', help='Directory to save results')
    args = parser.parse_args()

    framework = args.framework
    param = args.param
    data_dir = args.data_dir
    result_dir = args.result_dir

    datasets = ['cifar10', 'crisis_mmd', 'ku_har', 'crema_d']

    plt.rcParams.update({'font.size': 16})
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    def plot_for_datasets(group, fig_num):
        num_datasets = len(group)
        rows = 1 if num_datasets == 1 else 2
        cols = 1 if num_datasets == 1 else 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
        axes = axes.flatten()

        for ax, dataset_name in zip(axes, group):
            if dataset_name not in dataset_info:
                print(f"Dataset '{dataset_name}' not recognized.")
                continue

            model_name = dataset_info[dataset_name]['model']
            seeds = dataset_info[dataset_name]['seeds']
            param_values = dataset_info[dataset_name][param + 's']

            # Build mapping from (param_value, seed) to file paths
            file_mapping = defaultdict(list)
            for root, dirs, files in os.walk(data_dir):
                for filename in files:
                    if filename.endswith('.json'):
                        pattern = rf"{dataset_name}_{model_name}_lr_.*?_{param}_(\d+)_.*_seed_(\d+)_users_.*"
                        match = re.match(pattern, filename)
                        if match:
                            param_value, seed = match.groups()
                            seed = int(seed)
                            if seed not in seeds:
                                continue
                            filepath = os.path.join(root, filename)
                            file_mapping[(param_value, seed)].append(filepath)

            # Initialize data structures to collect training_loss over iterations
            loss_data = defaultdict(list)

            for (param_value, seed), filepaths in file_mapping.items():
                for filepath in filepaths:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        training_loss = data.get('training_loss', [])
                        if training_loss:
                            loss_data[param_value].append(training_loss)

            # Plot data for each param_value
            for param_value, color in zip(param_values, colors):
                if param_value in loss_data and loss_data[param_value]:
                    max_length = max(len(loss) for loss in loss_data[param_value])
                    loss_arrays = []
                    for loss in loss_data[param_value]:
                        if len(loss) < max_length:
                            loss.extend([loss[-1]] * (max_length - len(loss)))
                        loss_arrays.append(loss)
                    loss_arrays = np.array(loss_arrays)
                    avg_loss = np.mean(loss_arrays, axis=0)
                    ax.plot(avg_loss, label=f'{param}={param_value}', color=color)

            dataset_name_formatted = format_dataset_name(dataset_name)
            ax.set_title(dataset_name_formatted, fontsize=18)
            ax.set_xlabel('Communication Rounds', fontsize=18)
            ax.set_ylabel('Training Loss', fontsize=18)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(param_values), fontsize=18, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot for this group
        os.makedirs(result_dir, exist_ok=True)
        plot_file_png = os.path.join(result_dir, f'training_loss_{framework}_group{fig_num}_{param}.png')
        plot_file_pdf = os.path.join(result_dir, f'training_loss_{framework}_group{fig_num}_{param}.pdf')
        plt.savefig(plot_file_png)
        plt.savefig(plot_file_pdf)
        plt.close()

        print(f"Grouped training loss saved at {plot_file_png} and {plot_file_pdf}")

    def format_dataset_name(name):
        return {
            'cifar10': 'CIFAR10',
            'crema_d': 'CREMA-D',
            'crisis_mmd': 'CrisisMMD',
            'ku_har': 'KU-HAR'
        }.get(name, name)

    # Split the datasets into two groups
    group_1 = datasets[:2]  # First two datasets
    group_2 = datasets[2:]  # Third and fourth datasets

    # Plot the first group of datasets
    if group_1:
        plot_for_datasets(group_1, fig_num=1)

    # Plot the second group of datasets
    if group_2:
        plot_for_datasets(group_2, fig_num=2)

if __name__ == "__main__":
    main()
