#!/bin/bash
> log_base.txt
conda activate FedAPM
frameworks=("FedAPM" "FedAvg" "FedAlt" "FedSim" "FedProx")
datasets=( "crisis_mmd" "ku_har" "crema_d" "cifar10")
epochs=300
local_ep=3
mu=0.01
Lambda=1


declare -A dataset_seeds
dataset_seeds[hateful_memes]="4 5 6 7 8"
dataset_seeds[crisis_mmd]="1 3 4 6 7"
dataset_seeds[ku_har]="9 10 11 12 13"
dataset_seeds[crema_d]="10 11 12 13 14"
dataset_seeds[cifar10]="4 5 6 7 8"


for framework in "${frameworks[@]}"; do
  if [[ $framework == "FedAPM" || $framework == "FedAvg" || $framework == "FedAlt" || $framework == "FedSim" || $framework == "FedProx" ]]; then
    for dataset in "${datasets[@]}"; do
      seeds=${dataset_seeds[$dataset]}
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
        lr=0.1
      fi

      for seed in $seeds; do
        echo "Running experiment with the following configuration:" >> log_base.txt 2>&1
        echo "Framework: $framework" >> log_base.txt 2>&1
        echo "Partition: dir-label-skew" >> log_base.txt 2>&1
        echo "Dataset: $dataset" >> log_base.txt 2>&1
        echo "Model: $model" >> log_base.txt 2>&1
        echo "Num Users: $num_users" >> log_base.txt 2>&1
        echo "Frac Candidates: $frac_candidates" >> log_base.txt 2>&1
        echo "Frac: $frac" >> log_base.txt 2>&1
        echo "Layer Num: $layer_num" >> log_base.txt 2>&1
        echo "Learning Rate: $lr" >> log_base.txt 2>&1
        echo "Seed: $seed" >> log_base.txt 2>&1

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
                    --layer_num $layer_num 
                    # --layer_num $layer_num >> log_ku_har1.txt 2>&1
      done
    done
  fi
done

