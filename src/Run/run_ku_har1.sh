#!/bin/bash

> log_ku_har1.txt
conda activate FedAPM
frameworks=("FedAPM" "FedAvg" "FedAlt" "FedSim" "FedProx")
partitions=("q-label-skew" "dir-label-skew")
lr=1
epochs=300
local_ep=3
mu=0.01
Lambda=1
rho=0.01
for seed in 9 10 11 12 13; do
    for framework in "${frameworks[@]}"; do
      # if [[  $framework == "FedAPM" ]]; then
      if [[ $framework == "FedAPM" || $framework == "FedAvg" || $framework == "FedAlt" || $framework == "FedSim" || $framework == "FedProx" ]]; then
        # for dataset in hateful_memes crisis_mmd ku_har crema_d; do
        for dataset in ku_har; do
          if [[ $dataset == "crisis_mmd" ]]; then
            model="ImageTextClassifier"
            num_users=20
            frac=0.3
            frac_candidates="$frac"
            layer_num=8
            lr=0.5
          elif [[ $dataset == "ku_har" ]]; then
            model="HARClassifier"
            num_users=20
            frac=0.2
            frac_candidates="$frac"
            layer_num=12
            lr=0.5
          elif [[ $dataset == "crema_d" ]]; then
            model="MMActionClassifier"
            num_users=72
            frac=0.1
            frac_candidates="$frac"
            layer_num=14
            lr=1
          elif [[ $dataset == "cifar10" ]]; then
            model="CNN"
            num_users=20
            frac=0.3
            frac_candidates="$frac"
            layer_num=4
          fi

          echo "Running experiment with the following configuration:" >> log_ku_har1.txt 2>&1
          echo "Framework: $framework" >> log_ku_har1.txt 2>&1
          echo "Partition: dir-label-skew" >> log_ku_har1.txt 2>&1
          echo "Dataset: $dataset" >> log_ku_har1.txt 2>&1
          echo "Model: $model" >> log_ku_har1.txt 2>&1
          echo "Num Users: $num_users" >> log_ku_har1.txt 2>&1
          echo "Frac Candidates: $frac_candidates" >> log_ku_har1.txt 2>&1
          echo "Frac: $frac" >> log_ku_har1.txt 2>&1
          echo "Layer Num: $layer_num" >> log_ku_har1.txt 2>&1
          echo "Learning Rate: $lr" >> log_ku_har1.txt 2>&1
          echo "Seed: $seed" >> log_ku_har1.txt 2>&1


          python ./c1.py --framework $framework \
                      --partition q-label-skew \
                      --num_users $num_users \
                      --model $model \
                      --dataset $dataset \
                      --strategy random \
                      --frac_candidates $frac_candidates \
                      --frac $frac \
                      --lr $lr \
                      --seed $seed \
                      --epochs $epochs \
                      --local_ep $local_ep \
                      --mu $mu \
                      --Lambda $Lambda \
                      --rho $rho \
                      --layer_num $layer_num 
                      # --layer_num $layer_num >> log_ku_har1.txt 2>&1

        done
      fi
    done
  done
done

dataset_info = {
    'cifar10': {
        'model': 'CNN',
        'seeds': [4, 5, 6, 7, 8],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    },
    'crisis_mmd': {
        'model': 'ImageTextClassifier',
        'seeds': [1, 3, 4, 6, 7],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    },
    'ku_har': {
        'model': 'HARClassifier',
        'seeds': [9, 10, 11, 12, 13],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    },
    'crema_d': {
        'model': 'MMActionClassifier',
        'seeds': [10, 11, 12, 13, 14],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    }
}
