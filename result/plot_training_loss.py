import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import re
import sys

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot training loss for a dataset and seed.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--seed', type=int, required=True, help='Seed value')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Result files')
    args = parser.parse_args()

    dataset_name = args.dataset
    seed_value = args.seed
    data_dir = args.data_dir
    result_dir = args.result_dir
    

    # Mapping from dataset to model
    dataset_model_map = {
        'mnist': 'MLP',
        'cifar10': 'ResNet18',
        'hateful_memes': 'ImageTextClassifier',
        'crisis_mmd': 'ImageTextClassifier',
        'ku_har': 'HARClassifier',
        'crema_d': 'MMActionClassifier'
    }

    if dataset_name not in dataset_model_map:
        print(f"Model not found for dataset {dataset_name}")
        return

    model_name = dataset_model_map[dataset_name]

    frameworks = ['FedAPM', 'FedAvg', 'FedAlt', 'FedSim', 'FedProx']

    # Regular expression to parse filenames
    filename_pattern = re.compile(rf"^{dataset_name}_(.*?)_(.*?)_random_seed_{seed_value}_users_(\d+)_.*\.json$")

    # Build mapping from framework to file path
    file_mapping = {}
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.json'):
                match = filename_pattern.match(filename)
                if match:
                    model, framework, userval = match.groups()
                    if model != model_name:
                        continue
                    if framework not in frameworks:
                        continue
                    filepath = os.path.join(root, filename)
                    file_mapping[framework] = filepath

    if not file_mapping:
        print(f"No files found for dataset {dataset_name} and seed {seed_value}")
        return

    # Prepare result directory  
    os.makedirs(result_dir, exist_ok=True)

    # Plot training loss
    plt.figure(figsize=(14, 7))
    for framework in frameworks:
        if framework in file_mapping:
            filepath = file_mapping[framework]
            with open(filepath, 'r') as f:
                data = json.load(f)
                train_loss = data.get('training_loss', [])
                plt.plot(train_loss, label=f'{framework} Training Loss', linewidth=2)
    # plt.title(f'Training Loss Comparison for Seed {seed_value}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(result_dir, f"{dataset_name}_training_loss_seed_{seed_value}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Training loss plot saved at {plot_file}")
    
    

if __name__ == "__main__":
    main()
