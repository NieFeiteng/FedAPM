#!/bin/bash
> log_fedapm_experiments.txt
conda activate FedAPM

# 定义每个数据集的默认参数
declare -A default_lr
declare -A default_rho
declare -A default_frac
declare -A default_layer_num
declare -A default_num_users

# 从 default_params 获取默认参数和 layer_num, num_users
default_lr[crisis_mmd]=0.5
default_rho[crisis_mmd]=0.01
default_frac[crisis_mmd]=0.3
default_layer_num[crisis_mmd]=8
default_num_users[crisis_mmd]=20

default_lr[crema_d]=1.0
default_rho[crema_d]=0.01
default_frac[crema_d]=0.1
default_layer_num[crema_d]=14
default_num_users[crema_d]=72

default_lr[ku_har]=0.5
default_rho[ku_har]=0.01
default_frac[ku_har]=0.2
default_layer_num[ku_har]=12
default_num_users[ku_har]=20

default_lr[cifar10]=0.1
default_rho[cifar10]=0.01
default_frac[cifar10]=0.3
default_layer_num[cifar10]=4
default_num_users[cifar10]=20


declare -A dataset_seeds
dataset_seeds[hateful_memes]="4 5 6 7 8"
dataset_seeds[crisis_mmd]="1 3 4 6 7"
dataset_seeds[ku_har]="9 10 11 12 13"
dataset_seeds[crema_d]="10 11 12 13 14"
dataset_seeds[cifar10]="4 5 6 7 8"


declare -A dataset_info
dataset_info[cifar10]='CNN|4,5,6,7,8|1.0,0.5,0.2,0.1|0.1,0.05,0.02,0.01|0.5,0.3,0.2,0.1'
dataset_info[crisis_mmd]='ImageTextClassifier|1,3,4,6,7|1.0,0.5,0.2,0.1|0.1,0.05,0.02,0.01|0.5,0.3,0.2,0.1'
dataset_info[ku_har]='HARClassifier|9,10,11,12,13|1.0,0.5,0.2,0.1|0.1,0.05,0.02,0.01|0.5,0.3,0.2,0.1'
dataset_info[crema_d]='MMActionClassifier|10,11,12,13,14|1.0,0.5,0.2,0.1|0.1,0.05,0.02,0.01|0.5,0.3,0.2,0.1'

framework="FedAPM"
epochs=300
local_ep=3
mu=0.01
Lambda=1


run_experiment() {
    local dataset=$1
    local model=$2
    local seed=$3
    local lr=$4
    local rho=$5
    local frac=$6
    local layer_num=$7
    local num_users=$8

    echo "Running experiment for $dataset with seed=$seed, lr=$lr, rho=$rho, frac=$frac, layer_num=$layer_num, num_users=$num_users" >> log_fedapm_experiments.txt 2>&1

    python ./c1.py --framework $framework \
        --partition "q-label-skew" \
        --num_users "$num_users" \
        --model "$model" \
        --dataset "$dataset" \
        --strategy random \
        --frac_candidates "$frac" \
        --frac "$frac" \
        --lr "$lr" \
        --seed "$seed" \
        --epochs "$epochs" \
        --local_ep "$local_ep" \
        --mu "$mu" \
        --Lambda "$Lambda" \
        --rho "$rho" \
        --layer_num "$layer_num" >> log_fedapm_experiments.txt 2>&1
}


for dataset in "${!dataset_info[@]}"; do
    IFS='|' read -r model seeds lrs rhos fracs <<< "${dataset_info[$dataset]}"

    IFS=',' read -r -a seeds_array <<< "$seeds"
    IFS=',' read -r -a lrs_array <<< "$lrs"
    IFS=',' read -r -a rhos_array <<< "$rhos"
    IFS=',' read -r -a fracs_array <<< "$fracs"


    default_lr_value=${default_lr[$dataset]}
    default_rho_value=${default_rho[$dataset]}
    default_frac_value=${default_frac[$dataset]}
    default_layer_num_value=${default_layer_num[$dataset]}
    default_num_users_value=${default_num_users[$dataset]}

    for lr in "${lrs_array[@]}"; do
        for seed in "${seeds_array[@]}"; do
            run_experiment "$dataset" "$model" "$seed" "$lr" "$default_rho_value" "$default_frac_value" "$default_layer_num_value" "$default_num_users_value"
        done
    done

    for rho in "${rhos_array[@]}"; do
        for seed in "${seeds_array[@]}"; do
            run_experiment "$dataset" "$model" "$seed" "$default_lr_value" "$rho" "$default_frac_value" "$default_layer_num_value" "$default_num_users_value"
        done
    done

    for frac in "${fracs_array[@]}"; do
        for seed in "${seeds_array[@]}"; do
            run_experiment "$dataset" "$model" "$seed" "$default_lr_value" "$default_rho_value" "$frac" "$default_layer_num_value" "$default_num_users_value"
        done
    done
done
