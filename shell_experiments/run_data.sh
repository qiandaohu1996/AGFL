source D:/AGFL-main/shell_experiments/run.sh


# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist" "emnist_component4" "femnist" "cifar10" "cifar100")
DATA=("cifar100")



algo="FedAvg"

for dataset in "${DATA[@]}"
    do
        run_batch $dataset   
    done

algo="clustered"

for dataset in "${DATA[@]}"
    do
        run_batch $dataset 
    done


DATA=("cifar10" "cifar100")


algo="FuzzyFL"

fuzzy_m_list=("1.5" "1.6" "1.7")
 
for dataset in "${DATA[@]}"; do
    for m in "${fuzzy_m_list[@]}"; do
        # inner_dir="_pre_25_m_$m"
        run_batch $dataset --pre_rounds 25 --fuzzy_m "$m"
    done
done