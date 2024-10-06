#!/bin/bash

> log_crema_d1.txt
conda activate FedAPM
frameworks=("FedAPM" "FedAvg" "FedAlt" "FedSim" "FedProx")
partitions=("q-label-skew" "dir-label-skew")
lr=1
epochs=300
local_ep=3
mu=0.01
Lambda=1
rho=0.01
for seed in 10 11 12 13 14; do
  for framework in "${frameworks[@]}"; do
    if [[ $framework == "FedAPM" || $framework == "FedAvg" || $framework == "FedAlt" || $framework == "FedSim" || $framework == "FedProx" ]]; then
      # for dataset in hateful_memes crisis_mmd ku_har crema_d; do
      for dataset in crema_d; do
        if [[ $dataset == "hateful_memes" ]]; then
          model="ImageTextClassifier"
          num_users=20
          frac_candidates=0.3
          frac=0.3
          layer_num=8
        elif [[ $dataset == "crisis_mmd" ]]; then
          model="ImageTextClassifier"
          num_users=20
          frac_candidates=0.3
          frac=0.3
          layer_num=8
        elif [[ $dataset == "ku_har" ]]; then
          model="HARClassifier"
          num_users=63
          frac_candidates=0.1
          frac=0.1
          layer_num=20
        elif [[ $dataset == "crema_d" ]]; then
          model="MMActionClassifier"
          num_users=72
          frac_candidates=0.1
          frac=0.1
          layer_num=14
        fi

        echo "Running experiment with the following configuration:" >> log_crema_d1.txt 2>&1
        echo "Framework: $framework" >> log_crema_d1.txt 2>&1
        echo "Partition: q-label-skew" >> log_crema_d1.txt 2>&1
        echo "Dataset: $dataset" >> log_crema_d1.txt 2>&1
        echo "Model: $model" >> log_crema_d1.txt 2>&1
        echo "Num Users: $num_users" >> log_crema_d1.txt 2>&1
        echo "Frac Candidates: $frac_candidates" >> log_crema_d1.txt 2>&1
        echo "Frac: $frac" >> log_crema_d1.txt 2>&1
        echo "Layer Num: $layer_num" >> log_crema_d1.txt 2>&1
        echo "Learning Rate: $lr" >> log_crema_d1.txt 2>&1
        echo "Seed: $seed" >> log_crema_d1.txt 2>&1


        python ../c2.py --framework $framework \
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
                     --layer_num $layer_num >> log_crema_d1.txt 2>&1
      done
    fi
  done
done
