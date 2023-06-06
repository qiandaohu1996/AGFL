source D:/AGFL-main/shell_experiments/run.sh


# DATA=("emnist" emnist_compon4ent "femnist" "cifar10" "cifar100")
DATA=("shakespeare")

algo="FedProx"
 
for dataset in "${DATA[@]}"
do
run_gd $dataset --mu '0.5'
done
 