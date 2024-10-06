#!/bin/bash


# datasets=("mnist" "cifar10" "hateful_memes" "crisis_mmd" "ku_har" "crema_d")

datasets=("cifar10" "crisis_mmd" "ku_har" "crema_d")
data_dir="../save"


for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"

    result_dir="${data_dir}/../result/${dataset}_Result"
    mkdir -p "$result_dir"

    python generate_max_values_csv.py --dataset "$dataset" --data_dir "$data_dir" --result_dir "$result_dir"

    for seed in {1..15}
    do
        echo "Processing seed: $seed for dataset: $dataset"
        python plot_test_accuracy.py --dataset "$dataset" --seed "$seed" --data_dir "$data_dir" --result_dir "$result_dir"
        python plot_training_loss.py --dataset "$dataset" --seed "$seed" --data_dir "$data_dir" --result_dir "$result_dir"
    done
done
