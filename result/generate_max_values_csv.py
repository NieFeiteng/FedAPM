import os
import json
import csv
import re
import sys
from collections import defaultdict

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate CSV with maximum values for a dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Result files')
    args = parser.parse_args()

    dataset_name = args.dataset
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
    seeds = range(1, 16)

    # Regular expression to parse filenames
    filename_pattern = re.compile(rf"^{dataset_name}_(.*?)_(.*?)_random_seed_(\d+)_users_(\d+)_.*\.json$")

    # Build mapping from (seed, framework) to file path
    file_mapping = defaultdict(dict)
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.json'):
                match = filename_pattern.match(filename)
                if match:
                    model, framework, seedvalue, userval = match.groups()
                    seedvalue = int(seedvalue)
                    if model != model_name:
                        continue
                    if framework not in frameworks:
                        continue
                    if seedvalue not in seeds:
                        continue
                    filepath = os.path.join(root, filename)
                    file_mapping[(seedvalue, framework)] = filepath

    # Prepare result directory
    os.makedirs(result_dir, exist_ok=True)
    csv_file = os.path.join(result_dir, f"{dataset_name}_avg_max_values.csv")

    # Write to CSV
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['seed', 'framework', 'max_test_acc', 'max_f1_score', 'max_auc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for seed in seeds:
            for framework in frameworks:
                key = (seed, framework)
                if key in file_mapping:
                    filepath = file_mapping[key]
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        test_acc = data.get('test_acc', [])
                        f1_score = data.get('f1_score', [])
                        auc = data.get('auc', [])
                        max_test_acc = max(test_acc) if test_acc else None
                        max_f1_score = max(f1_score) if f1_score else None
                        max_auc = max(auc) if auc else None
                        writer.writerow({
                            'seed': seed,
                            'framework': framework,
                            'max_test_acc': max_test_acc,
                            'max_f1_score': max_f1_score,
                            'max_auc': max_auc
                        })
                else:
                    # Handle missing files
                    writer.writerow({
                        'seed': seed,
                        'framework': framework,
                        'max_test_acc': None,
                        'max_f1_score': None,
                        'max_auc': None
                    })

    print(f"CSV file generated at {csv_file}")

if __name__ == "__main__":
    main()
