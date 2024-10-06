
from seed_config import dataset_info




import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Generate training loss plot for FedAPM with varying lr, frac, and rho.')
    parser.add_argument('--datasets', nargs=2, required=True, help='Two dataset names')
    parser.add_argument('--data_dir', type=str, default='../save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='./', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir

    # Set font parameters
    plt.rcParams.update({'font.size': 16})

    # Define the colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Default parameter values for each dataset (provided by you)
    default_params = {
        'crisis_mmd': {'lr': 0.5, 'rho': 0.01, 'frac': 0.3},
        'crema_d': {'lr': 1.0, 'rho': 0.01, 'frac': 0.1},
        'ku_har': {'lr': 0.5, 'rho': 0.01, 'frac': 0.2},
        'cifar10': {'lr': 0.1, 'rho': 0.01, 'frac': 0.3}
    }

    def plot_metric_varying_param(param, fig_name_suffix):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))  # Two subplots for two datasets
        axes = axes.flatten()  # Flatten axes array for easier iteration

        for ax, dataset_name in zip(axes, datasets):
            if dataset_name not in dataset_info or dataset_name not in default_params:
                print(f"Dataset '{dataset_name}' not recognized.")
                continue

            seeds = dataset_info[dataset_name]['seeds']
            param_values = dataset_info[dataset_name][param + 's']

            # Get default values for other parameters
            default_lr = default_params[dataset_name]['lr']
            default_rho = default_params[dataset_name]['rho']
            default_frac = default_params[dataset_name]['frac']

            loss_data = defaultdict(list)

            # Iterate over param values
            for param_value, color in zip(param_values, colors):
                # Build filename pattern
                # We need to vary param_value, and fix other parameters
                if param == 'lr':
                    lr = param_value
                    rho = default_rho
                    frac = default_frac
                elif param == 'rho':
                    lr = default_lr
                    rho = param_value
                    frac = default_frac
                elif param == 'frac':
                    lr = default_lr
                    rho = default_rho
                    frac = param_value
                else:
                    continue  # Should not reach here

                # Iterate over seeds
                for seed in seeds:
                    # Match filenames using the updated pattern
                    filename_pattern = re.compile(
                        rf"^{dataset_name}_.*_FedAPM_random_lr_{lr}_frac_{frac}_seed_{seed}_users_.*_rho_{rho}_.*\.json"
                    )

                    # Search for matching files
                    # print(f"Looking for files matching pattern: {filename_pattern.pattern}")
                    for root, _, files in os.walk(data_dir):
                        for filename in files:
                            if filename_pattern.match(filename):
                                filepath = os.path.join(root, filename)
                                # print(f"Reading file: {filepath}")  # Print the filename being read
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    training_loss = data.get('training_loss', [])
                                    if training_loss:
                                        loss_data[param_value].append(training_loss)

                # Compute average loss across seeds
                if param_value in loss_data and loss_data[param_value]:
                    max_length = max(len(loss) for loss in loss_data[param_value])
                    loss_arrays = []
                    for loss in loss_data[param_value]:
                        if len(loss) < max_length:
                            loss.extend([loss[-1]] * (max_length - len(loss)))
                        loss_arrays.append(loss)
                    loss_arrays = np.array(loss_arrays)
                    avg_loss = np.mean(loss_arrays, axis=0)
                    
                    # ax.plot(avg_loss, label=f'{param}={param_value}', color=color)
                    if param == 'frac':
                        label_param = '$|\mathcal{S}^t|/m$'
                    elif param == 'rho':
                        label_param = '$\\rho$'
                    else:
                        label_param = param

                    ax.plot(avg_loss, label=f'{label_param}={param_value}', color=color)
                    



            dataset_name_formatted = dataset_name.replace('_', '-').upper()
            if dataset_name_formatted == 'CIFAR10':
                dataset_name_formatted = 'CIFAR10'
            elif dataset_name_formatted == 'CREMA-D':
                dataset_name_formatted = 'CREMA-D'
            elif dataset_name_formatted == 'CRISIS-MMD':
                dataset_name_formatted = 'CrisisMMD'
            elif dataset_name_formatted == 'KU-HAR':
                dataset_name_formatted = 'KU-HAR'            
            
            
            # ax.set_title(f'{dataset_name_formatted} - {param}', fontsize=18)
            ax.set_title(f'{dataset_name_formatted}', fontsize=18)
            ax.set_xlabel('Communication Rounds', fontsize=16)
            ax.set_ylabel('Training Loss', fontsize=16)

        # Adjust the legend to be larger and more readable
        handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', ncol=len(param_values), fontsize=18, frameon=False)

# Add bbox_to_anchor to move the legend up
        fig.legend(handles, labels, loc='upper center', ncol=len(param_values), fontsize=18, frameon=False, bbox_to_anchor=(0.5, 1.03))



        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(result_dir, exist_ok=True)

        # Save the figure, include dataset names in the filename
        fig_name = f'{datasets[0]}_{datasets[1]}_FedAPM_loss_varying_{fig_name_suffix}.png'
        plt.savefig(os.path.join(result_dir, fig_name))

        fig_name_pdf = f'{datasets[0]}_{datasets[1]}_FedAPM_loss_varying_{fig_name_suffix}.pdf'
        plt.savefig(os.path.join(result_dir, fig_name_pdf))

        plt.close()

    # Plot loss varying by lr, frac, and rho
    plot_metric_varying_param('lr', 'lr')
    plot_metric_varying_param('frac', 'frac')
    plot_metric_varying_param('rho', 'rho')

    print("Plots generated successfully.")

if __name__ == "__main__":
    main()
