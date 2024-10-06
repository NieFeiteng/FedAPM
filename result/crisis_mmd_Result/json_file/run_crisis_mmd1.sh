#!/bin/bash

cd /home/nft99/FedAPM/pFedADMM1/src/Run
> log_crisis_mmd1.txt

conda activate FedAPM

frameworks=("FedAPM" "FedAvg" "FedAlt" "FedSim" "FedProx")
partitions=("q-label-skew" "dir-label-skew")
lr=0.5
epochs=10
local_ep=3
mu=0.01
Lambda=1
rho=0.01
epochs=10
local_ep=3
mu=0.01
Lambda=1
rho=0.01
# for seed in {2..5}; do
for seed in 2; do
  for framework in "${frameworks[@]}"; do
    if [[ $framework == "FedAPM" || $framework == "FedAvg" || $framework == "FedAlt" || $framework == "FedSim" || $framework == "FedProx" ]]; then
      # for dataset in hateful_memes crisis_mmd ku_har crema_d; do
      for dataset in crisis_mmd ; do
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

        echo "Running experiment with the following configuration:" >> log_crisis_mmd1.txt 2>&1
        echo "Framework: $framework" >> log_crisis_mmd1.txt 2>&1
        echo "Partition: q-label-skew" >> log_crisis_mmd1.txt 2>&1
        echo "Dataset: $dataset" >> log_crisis_mmd1.txt 2>&1
        echo "Model: $model" >> log_crisis_mmd1.txt 2>&1
        echo "Num Users: $num_users" >> log_crisis_mmd1.txt 2>&1
        echo "Frac Candidates: $frac_candidates" >> log_crisis_mmd1.txt 2>&1
        echo "Frac: $frac" >> log_crisis_mmd1.txt 2>&1
        echo "Layer Num: $layer_num" >> log_crisis_mmd1.txt 2>&1
        echo "Learning Rate: $lr" >> log_crisis_mmd1.txt 2>&1
        echo "Seed: $seed" >> log_crisis_mmd1.txt 2>&1


        python ../c4.py --framework $framework \
                     --partition q-label-skew \
                     --num_users $num_users \
                     --model $model \
                     --dataset $dataset \
                     --strategy random \
                     --frac_candidates $frac_candidates \
                     --frac $frac \
                     --lr $lr \
                     --seed $seed \
                     --layer_num $layer_num >> log_crisis_mmd1.txt 2>&1
      done
    fi
  done
done
