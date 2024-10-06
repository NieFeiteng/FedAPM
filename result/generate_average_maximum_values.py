import os
import json
import re
import argparse
import csv
from collections import defaultdict
from seed_config import dataset_info  # Ensure this module exists and is correctly formatted
import numpy as np
def main():
    parser = argparse.ArgumentParser(description='Calculate and save average maximum values of test_acc, f1_score, and auc for multiple datasets and frameworks for specified seeds.')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
    parser.add_argument('--data_dir', type=str, default='save', help='Directory containing JSON files')
    parser.add_argument('--result_dir', type=str, default='save', help='Directory to save results')
    args = parser.parse_args()

    datasets = args.datasets
    data_dir = args.data_dir
    result_dir = args.result_dir

    # Frameworks to be processed
    frameworks = ['FedAPM','FedAvg', 'FedAlt', 'FedSim', 'FedProx']

    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Prepare the CSV file
    csv_file = os.path.join(result_dir, 'average_max_metrics.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        

        

        fieldnames = ['Dataset', 'Framework', 'Average_Max_Test_Acc', 'Acc_std', 'Average_Max_F1_Score','F1_std', 'Average_Max_AUC','AUC_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name in datasets:
            if dataset_name not in dataset_info:
                print(f"Dataset '{dataset_name}' not recognized. Skipping.")
                continue

            model_name = dataset_info[dataset_name]['model']
            seeds = dataset_info[dataset_name]['seeds']

            # Regular expression to parse filenames
            filename_pattern = re.compile(rf"^{dataset_name}_{model_name}_(.*?)_random_seed_(\d+)_users_(\d+)_.*\.json$")

            # For each framework, calculate the max test_acc, f1_score, and auc across specified seeds
            for framework in frameworks:
                max_test_acc = []
                max_f1_score = []
                max_auc = []

                for seed in seeds:
                    for root, dirs, files in os.walk(data_dir):
                        for filename in files:
                            if filename.endswith('.json'):
                                match = filename_pattern.match(filename)
                                if match:
                                    matched_framework, seedvalue, userval = match.groups()
                                    seedvalue = int(seedvalue)
                                    if seedvalue != seed or matched_framework != framework:
                                        continue
                                    filepath = os.path.join(root, filename)
                                    try:
                                        with open(filepath, 'r') as f:
                                            data = json.load(f)
                                            test_acc = data.get('test_acc', [])
                                            f1_score = data.get('f1_score', [])
                                            auc = data.get('auc', [])
                                            if test_acc:
                                                max_test_acc.append(max(test_acc))
                                            if f1_score:
                                                max_f1_score.append(max(f1_score))
                                            if auc:
                                                max_auc.append(max(auc))
                                    except json.JSONDecodeError:
                                        print(f"Error decoding JSON from file: {filepath}. Skipping.")
                                    except Exception as e:
                                        print(f"Unexpected error processing file: {filepath}. Error: {e}. Skipping.")

                # Calculate the average of the maximum values across seeds
                avg_max_test_acc = sum(max_test_acc) / len(max_test_acc) if max_test_acc else 0
                std_acc = np.std(max_test_acc)
                avg_max_f1_score = sum(max_f1_score) / len(max_f1_score) if max_f1_score else 0
                std_f1 = np.std(max_f1_score)
                avg_max_auc = sum(max_auc) / len(max_auc) if max_auc else 0
                std_auc = np.std(max_auc)

                # Write the results to the CSV file, formatting values to three decimal places
                writer.writerow({
                    'Dataset': dataset_name,
                    'Framework': framework,
                    'Average_Max_Test_Acc': f'{avg_max_test_acc:.3f}',
                    'Acc_std': f'{std_acc:.3f}',
                    'Average_Max_F1_Score': f'{avg_max_f1_score:.3f}',
                    'F1_std': f'{std_f1:.3f}',
                    'Average_Max_AUC': f'{avg_max_auc:.3f}',
                    'AUC_std': f'{std_auc:.3f}'

                })

    print(f"Results saved in {csv_file}")

if __name__ == "__main__":
    main()
