#!/bin/bash

# Set data directory and result directory
DATA_DIR="../save"
RESULT_DIR="./Figures"

datasets=("cifar10" "crisis_mmd" "ku_har" "crema_d")

python generate_grouped_norm_bar_chart.py --datasets "${datasets[@]}" --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR"
python generate_training_loss_subplots.py --datasets "${datasets[@]}" --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR"
python generate_test_acc_subplots.py --datasets "${datasets[@]}" --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR"
python generate_average_maximum_values.py --datasets "${datasets[@]}" --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR"
python del_avg_metrics.py 


# Datasets to plot
dataset1='ku_har'
dataset2='crema_d'   

# Call the Python script
python var_para.py --datasets "$dataset1" "$dataset2" --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR"

# Datasets to plot
dataset1='cifar10'
dataset2='crisis_mmd'  

# Call the Python script
python var_para.py --datasets "$dataset1" "$dataset2" --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR"