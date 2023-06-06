source D:/AGFL-main/shell_experiments/run.sh

# DATA=("emnist_pathologic_cl20" "emnist_component4" "femnist" "cifar10" "cifar100")
DATA=("femnist")

algo="FedEM" 
inner_dir="_samp_0.5_m_3"

optimizer_name="sgd"

for dataset in "${DATA[@]}"
    do
        run $dataset  --sampling_rate 0.5 
    done
