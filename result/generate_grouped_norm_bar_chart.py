import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from seed_config import dataset_info  # Ensure this module exists and is correctly formatted

def main():
    parser = argparse.ArgumentParser(description='Generate grouped norm bar chart for multiple datasets.')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir

    # Define frameworks and their corresponding colors
    frameworks = ['FedAPM','FedAlt', 'FedSim', 'FedAvg', 'FedProx']
    # colors = ['#C8EFFE', '#AD8EDA', '#F9CDD9', '#FFD15B', '#6FD7A3']
    colors = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#000000']
    hatches = ['/', '\\', '|', '-']

    # Set font parameters
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16})

    # Prepare data structure to hold average norm values
    norm_values = defaultdict(dict)  # dataset -> framework -> avg_norm

    for dataset_name in datasets:
        if dataset_name not in dataset_info:
            print(f"Dataset '{dataset_name}' not recognized. Skipping.")
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

        # Initialize data structures to collect norm values
        dataset_norm_data = defaultdict(list)  # framework -> list of norm values

        # Iterate over files and collect norm data
        for (framework, seed), filepaths in file_mapping.items():
            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        norm_values_list = data.get('norm', [])
                        if norm_values_list:
                            # Extend the list with all norm values from this file
                            dataset_norm_data[framework].extend(norm_values_list)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}. Skipping.")
                except Exception as e:
                    print(f"Unexpected error processing file: {filepath}. Error: {e}. Skipping.")

        # Calculate the average norm value for each framework in this dataset
        for framework in frameworks:
            norms = dataset_norm_data[framework]
            norms = norms[:30]
            if norms:
                avg_norm = sum(norms) / len(norms)
                norm_values[dataset_name][framework] = avg_norm
            else:
                norm_values[dataset_name][framework] = 0  # Handle as needed

    plt.figure(figsize=(14, 5))  # Adjust figure size as needed

    # Number of datasets and frameworks
    num_datasets = len(datasets)
    num_frameworks = len(frameworks)
    bar_width = 0.15  # Adjusted bar width to fit 5 bars

    # X locations for the datasets
    index = np.arange(num_datasets)  

    # Adjust index for each framework, space them evenly around the main index
    offsets = np.linspace(-2 * bar_width, 2 * bar_width, num_frameworks)

    for i, (framework, color) in enumerate(zip(frameworks, colors)):
        # Extract average norm for each dataset for the current framework
        avg_norms = [norm_values[dataset].get(framework, 0) for dataset in datasets]
        
        # Plot bars with adjusted offsets for the framework
        bars = plt.bar(index + offsets[i], avg_norms, bar_width, label=framework, color=color, 
                    edgecolor='black', hatch=hatches[i % len(hatches)])  # Apply hatching patterns

        # Add text annotations on top of each bar
        for bar in bars:
            height = bar.get_height()

            # Conditional formatting based on the value of height
            # if height < 0.099:
            formatted_height = f'{height:.3f}'.lstrip('0')  # Remove leading zero for values less than 0.1
            # else:
            #     formatted_height = f'{height:.2f}'  # Two decimal places for values >= 0.1

            # Add a small vertical offset to ensure the text is not overlapping the bar
            plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                    formatted_height, ha='center', va='bottom', fontsize='smaller')

    # Add additional labels or styling as needed
    # plt.legend(title='Frameworks', fontsize=18)
    plt.legend(fontsize=20)

    plt.ylabel('$||u_i-u||$', fontsize=20)
    plt.tick_params(axis='x', which='both', length=0)
    dataset_names = ['CIFAR10', 'CrisisMMD','KU-HAR', 'CREMA-D']
    plt.xticks(index, dataset_names, fontsize=20)
    plt.yticks(fontsize=20)

    # Display or save the figure
    plt.tight_layout()
    plt.show()  # or plt.savefig('your_figure.png') to save the figure

    # Save the plot
    os.makedirs(result_dir, exist_ok=True)

    # Save the plot as PNG
    plot_file_png = os.path.join(result_dir, 'grouped_norm_bar_chart.png')
    plt.savefig(plot_file_png)

    # Save the plot as PDF
    plot_file_pdf = os.path.join(result_dir, 'grouped_norm_bar_chart.pdf')
    plt.savefig(plot_file_pdf)

    plt.close()

    print(f"Grouped norm bar chart saved at {plot_file_png} and {plot_file_pdf}")



if __name__ == "__main__":
    main()
