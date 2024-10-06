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
    parser = argparse.ArgumentParser(description='Generate training loss plot with subplots for multiple datasets.')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir

    # Define frameworks
    frameworks = ['FedAPM', 'FedAvg', 'FedAlt', 'FedSim', 'FedProx']

    # Set font parameters
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16})

    # Define the colors
    # colors = ['#C8EFFE', '#F9CDD9', '#FFD15B', '#6FD7A3', '#F68E64']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown','tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Prepare the figure with 2 rows and 2 columns of subplots
    num_datasets = len(datasets)
    rows = 2  # 2 rows
    cols = 2  # 2 columns
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))  # Adjust the size as needed
    axes = axes.flatten()  # Flatten axes array for easier iteration

    # Iterate over datasets and create subplots
    for ax, dataset_name in zip(axes, datasets):
        if dataset_name not in dataset_info:
            print(f"Dataset '{dataset_name}' not recognized.")
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

        # Initialize data structures to collect training_loss over iterations
        loss_data = defaultdict(list)  # framework -> list of training_loss arrays

        # Iterate over files and collect training_loss
        for (framework, seed), filepaths in file_mapping.items():
            for filepath in filepaths:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    training_loss = data.get('training_loss', [])
                    if training_loss:
                        loss_data[framework].append(training_loss)

        # Plot data for each framework
        for framework, color in zip(frameworks, colors):
            if framework in loss_data and loss_data[framework]:
                # Pad shorter arrays with last value to match lengths
                max_length = max(len(loss) for loss in loss_data[framework])
                loss_arrays = []
                for loss in loss_data[framework]:
                    if len(loss) < max_length:
                        loss.extend([loss[-1]] * (max_length - len(loss)))
                    loss_arrays.append(loss)
                loss_arrays = np.array(loss_arrays)
                avg_loss = np.mean(loss_arrays, axis=0)
                ax.plot(avg_loss, label=framework, color=color)
        if dataset_name == 'cifar10':
            dataset_name = 'CIFAR10'
        elif dataset_name == 'crema_d':
            dataset_name = 'CREMA-D'
        elif dataset_name == 'crisis_mmd':
            dataset_name = 'CrisisMMD'
        elif dataset_name == 'ku_har':
            dataset_name = 'KU-HAR'
        ax.set_title(dataset_name, fontsize=18)
        ax.set_xlabel('Communication Rounds', fontsize=18)
        ax.set_ylabel('Training Loss', fontsize=18)

    # Create a single shared legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(frameworks), fontsize=18, frameon=False)

    # Layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to fit the legend on top

    # Save the plot
    os.makedirs(result_dir, exist_ok=True)

    
    # Save the plot as PNG
    plot_file_png = os.path.join(result_dir, 'training_loss_subplots.png')
    plt.savefig(plot_file_png)

    # Save the plot as PDF
    plot_file_pdf = os.path.join(result_dir, 'training_loss_subplots.pdf')
    plt.savefig(plot_file_pdf)

    plt.close()

    print(f"Grouped training loss saved at {plot_file_png} and {plot_file_pdf}")
    

if __name__ == "__main__":
    main()
