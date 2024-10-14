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

def smooth_curve(data, window_size=10):
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, mode='valid')

def main():
    parser = argparse.ArgumentParser(description='Generate training loss plot with subplots for multiple datasets.')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir

    frameworks = ['FedAPM', 'FedAvg', 'FedAlt', 'FedSim', 'FedProx']

    plt.rcParams.update({'font.size': 16})

    colors = ['#000000', '#000000', '#000000', '#000000', '#000000']
    markers = ['o', 's', '^', 'x', 'v'] 
    linestyles = ['-', '--', '-.', ':', '-'] 

    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 4.5))  # Single row with as many columns as datasets
    axes = axes.flatten()  # Flatten axes array for easier iteration

    for ax, dataset_name in zip(axes, datasets):
        if dataset_name not in dataset_info:
            print(f"Dataset '{dataset_name}' not recognized.")
            continue

        model_name = dataset_info[dataset_name]['model']
        seeds = dataset_info[dataset_name]['seeds']

        filename_pattern = re.compile(rf"^{dataset_name}_{model_name}_(.*?)_random_seed_(\d+)_users_(\d+)_.*\.json$")

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

        loss_data = defaultdict(list)

        for (framework, seed), filepaths in file_mapping.items():
            for filepath in filepaths:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    training_loss = data.get('training_loss', [])
                    if training_loss:
                        loss_data[framework].append(training_loss)

        for framework, color, marker, linestyle in zip(frameworks, colors, markers, linestyles):
            if framework in loss_data and loss_data[framework]:
                max_length = max(len(loss) for loss in loss_data[framework])
                loss_arrays = []
                for loss in loss_data[framework]:
                    if len(loss) < max_length:
                        loss.extend([loss[-1]] * (max_length - len(loss)))
                    loss_arrays.append(loss)
                loss_arrays = np.array(loss_arrays)
                avg_loss = np.mean(loss_arrays, axis=0)
                avg_loss = smooth_curve(avg_loss)
                ax.plot(avg_loss, label=framework, color=color, marker=marker, markevery=58, markersize=12, markerfacecolor='none', markeredgecolor=color, linestyle=linestyle)

        formatted_name = dataset_name.upper().replace('_', '-')
        ax.set_title(formatted_name, fontsize=18)
        ax.set_xlabel('Communication Rounds', fontsize=20)
        ax.set_ylabel('Training Loss', fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(frameworks), fontsize=18, frameon=False, bbox_to_anchor=(0.5, 1.03))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(result_dir, exist_ok=True)
    fig_name = f'{"_".join(datasets)}_training_loss.png'
    fig_name_pdf = f'{"_".join(datasets)}_training_loss.pdf'

    plt.savefig(os.path.join(result_dir, fig_name))
    plt.savefig(os.path.join(result_dir, fig_name_pdf))

    plt.close()

    print(f"Grouped training loss saved at {os.path.join(result_dir, fig_name)} and {os.path.join(result_dir, fig_name_pdf)}")

if __name__ == "__main__":
    main()
