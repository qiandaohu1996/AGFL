#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source D:/AGFL-main/shell_experiments/run.sh
# DATA=("emnist" "emnist_component4" "femnist" "cifar10" "cifar100")
# DATA=("emnist_component4" "femnist" "cifar10")
DATA=("emnist")

 

# algos=("FedAvg" "local" "FedProx"  "APFL" "FedEM"  "pFedMe" "clustered" "FuzzyFL" "AGFL" "L2SGD" "AFL" "FFL")
 
# algo="APFL" 
algo="AGFL" 


for dataset in "${DATA[@]}"
    do
        run $dataset  --alpha 0.5  --adaptive_alpha
    done


