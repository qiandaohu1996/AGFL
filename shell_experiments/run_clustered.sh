source D:/AGFL-main/shell_experiments/run.sh

DATA=("emnist" "emnist_compon4ent" "femnist" "cifar10" "cifar100")
# DATA=("emnist" )
# dataset="shakespeare"

algo="clustered"

for dataset in "${DATA[@]}"
    do
        run_batch $dataset 
    done


