import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from seed_config import dataset_info

def smooth_curve(data, window_size=10):
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, mode='valid')

def main():
    parser = argparse.ArgumentParser(description='Generate training loss plot with insets for specific datasets.')
    parser.add_argument('--datasets', nargs=4, required=True, help='Four dataset names')
    parser.add_argument('--data_dir', type=str, default='../save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='./', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir

    plt.rcParams.update({'font.size': 16})
    colors = ['#000000', '#000000', '#000000', '#000000', '#000000']
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'x', 'v']
    default_params = {
        'crisis_mmd': {'lr': 0.5, 'rho': 0.01, 'frac': 0.3},
        'crema_d': {'lr': 1.0, 'rho': 0.01, 'frac': 0.1},
        'ku_har': {'lr': 0.5, 'rho': 0.01, 'frac': 0.2},
        'cifar10': {'lr': 0.1, 'rho': 0.01, 'frac': 0.3}
    }

    def plot_metric_varying_param(param, fig_name_suffix):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))  # Four subplots for four datasets
        for ax, dataset_name in zip(axes, datasets):
            if dataset_name not in dataset_info or dataset_name not in default_params:
                print(f"Dataset '{dataset_name}' not recognized.")
                continue

            seeds = dataset_info[dataset_name]['seeds']
            param_values = dataset_info[dataset_name][param + 's']
            default_lr = default_params[dataset_name]['lr']
            default_rho = default_params[dataset_name]['rho']
            default_frac = default_params[dataset_name]['frac']
            loss_data = defaultdict(list)

            for param_value, color, linestyle, marker in zip(param_values, colors, linestyles, markers):
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

                for seed in seeds:
                    filename_pattern = re.compile(
                        rf"^{dataset_name}_.*_FedAPM_random_lr_{lr}_frac_{frac}_seed_{seed}_users_.*_rho_{rho}_.*\.json"
                    )
                    for root, _, files in os.walk(data_dir):
                        for filename in files:
                            if filename_pattern.match(filename):
                                filepath = os.path.join(root, filename)
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    training_loss = data.get('training_loss', [])
                                    if training_loss:
                                        loss_data[param_value].append(training_loss)

                if param_value in loss_data and loss_data[param_value]:
                    max_length = max(len(loss) for loss in loss_data[param_value])
                    loss_arrays = []
                    for loss in loss_data[param_value]:
                        if len(loss) < max_length:
                            loss.extend([loss[-1]] * (max_length - len(loss)))
                        loss_arrays.append(loss)
                    loss_arrays = np.array(loss_arrays)
                    avg_loss = np.mean(loss_arrays, axis=0)
                    label_param = '$|\mathcal{S}^t|/m$' if param == 'frac' else '$\\rho$' if param == 'rho' else param
                    avg_loss = smooth_curve(avg_loss)
                    ax.plot(avg_loss, label=f'{label_param}={param_value}', color=color, linestyle=linestyle, marker=marker, markevery=58, markersize=12, markerfacecolor='none', markeredgecolor=color)

            # Add inset only for CIFAR10 and CrisisMMD
            if (dataset_name == 'cifar10' and param == 'rho'):
                ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper right")
                ax_inset.set_xlim(40, 70)  # Adjust zoom region as needed
                ax_inset.set_ylim(min(avg_loss[30:70]) - 0.01, max(avg_loss[30:70]) + 0.01)
                for param_value, color, linestyle, marker in zip(param_values, colors, linestyles, markers):
                    if param_value in loss_data and loss_data[param_value]:
                        avg_loss = np.mean(loss_data[param_value], axis=0)
                        avg_loss = smooth_curve(avg_loss)
                        ax_inset.plot(avg_loss, label=f'{label_param}={param_value}', color=color, linestyle=linestyle, marker=marker, markevery=58, markersize=8, markerfacecolor='none', markeredgecolor=color)
                ax_inset.tick_params(axis='both', which='major', labelsize=14)
                
            # Add inset only for CIFAR10 and CrisisMMD
            if (dataset_name == 'crisis_mmd' and param == 'rho'):
                ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper right")
                ax_inset.set_xlim(40, 80 )  # Adjust zoom region as needed
                ax_inset.set_ylim(min(avg_loss[40:80]) - 0.01, max(avg_loss[40:80]) + 0.01)
                for param_value, color, linestyle, marker in zip(param_values, colors, linestyles, markers):
                    if param_value in loss_data and loss_data[param_value]:
                        avg_loss = np.mean(loss_data[param_value], axis=0)
                        avg_loss = smooth_curve(avg_loss)
                        ax_inset.plot(avg_loss, label=f'{label_param}={param_value}', color=color, linestyle=linestyle, marker=marker, markevery=58, markersize=8, markerfacecolor='none', markeredgecolor=color)
                ax_inset.tick_params(axis='both', which='major', labelsize=14)

            ax.set_title(dataset_name.upper().replace('_', '-'), fontsize=18)
            ax.set_xlabel('Communication Rounds', fontsize=16)
            ax.set_ylabel('Training Loss', fontsize=16)

        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=len(param_values), frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(result_dir, exist_ok=True)

        fig_name = f'{"_".join(datasets)}_FedAPM_loss_varying_{fig_name_suffix}.png'
        fig_name_pdf =f'{"_".join(datasets)}_FedAPM_loss_varying_{fig_name_suffix}.pdf'
        plt.savefig(os.path.join(result_dir, fig_name_pdf))

        plt.savefig(os.path.join(result_dir, fig_name))
        plt.close()

    plot_metric_varying_param('lr', 'lr')
    plot_metric_varying_param('frac', 'frac')
    plot_metric_varying_param('rho', 'rho')

    print("Plots generated successfully.")

if __name__ == "__main__":
    main()
