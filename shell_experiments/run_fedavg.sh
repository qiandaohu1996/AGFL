source D:/AGFL-main/shell_experiments/run.sh


DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist")


algo="FedAvg"

for dataset in "${DATA[@]}"
    do
        run_batch $dataset   
    done
