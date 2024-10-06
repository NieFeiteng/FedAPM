#!/bin/bash

cd /home/nft99/FedAPM/pFedADMM1/src/Run
conda activate FedAPM
> log_crema_d1.txt
conda activate FedAPM
frameworks=("FedAPM" "FedAvg" "FedAlt" "FedSim" "FedProx")
# frameworks=("FedAPM" )
partitions=("q-label-skew" "dir-label-skew")
lr=1

for seed in {1..7}; do
# for seed in 1; do
  for framework in "${frameworks[@]}"; do
    # if [[ $framework == "FedAPM" || $framework == "FedAvg" || $framework == "FedAlt" || $framework == "FedSim" || $framework == "FedProx" ]]; then
      if [[ $framework == "FedAPM" || $framework == "FedAlt" || $framework == "FedSim" ]]; then
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
          # num_users=63
          num_users=20
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


        python ../c3.py --framework $framework \
                     --partition q-label-skew \
                     --num_users $num_users \
                     --model $model \
                     --dataset $dataset \
                     --strategy random \
                     --frac_candidates $frac_candidates \
                     --frac $frac \
                     --lr $lr \
                     --seed $seed \
                     --layer_num $layer_num >> log_crema_d1.txt 2>&1
      done
    fi
  done
done
