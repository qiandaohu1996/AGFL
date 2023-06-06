source D:/AGFL-main/shell_experiments/run.sh


# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist" "emnist_component4" "femnist" "cifar10" "cifar100")
dataset="cifar100_s0.25"

# algo="FedAvg"

# run $dataset  --locally_tune_clients  

# algo="local"

# run $dataset  

algo="clustered"

run_batch $dataset   

# algo="APFL"

# run $dataset  --alpha 0.5  --adaptive_alpha

algo="FedProx"
  
run_batch $dataset   --mu '0.1'

# algo="pFedMe"

# run $dataset   --mu '0.1'

# algo="FedProx"

# run $dataset   --mu '0.5'
 
# algo="pFedMe"

# run $dataset   --mu '0.5'

algo="FedEM"

run_batch $dataset  
