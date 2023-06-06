source D:/AGFL-main/shell_experiments/run.sh

# DATA=("emnist" "femnist" "cifar10" "cifar100_s0.25")
# DATA=("emnist" )

# DATA=("emnist_pathologic_cl20" "emnist_component4")

DATA=("cifar100_s0.25")

algo="FuzzyFL"

# fuzzy_m_list=("1.55" "1.65"  "1.75")
fuzzy_m_list=("1.6")
 

for dataset in "${DATA[@]}"; do
    for m in "${fuzzy_m_list[@]}"; do
        # inner_dir="_pre_25_m_$m"
        run_batch $dataset --pre_rounds 25 --fuzzy_m "$m"
    done
done