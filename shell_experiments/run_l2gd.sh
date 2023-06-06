source D:/AGFL-main/shell_experiments/run.sh

DATA=("emnist_pathologic_cl20" "emnist_component4" "femnist" "cifar10" "cifar100")

algos=("L2SGD" "FedProx")

for dataset in "${DATA[@]}"
    for algo in "${algos[@]}"
        do
            run $dataset  --comm_prob 0.2  --mu 0.1 
        done
    done

