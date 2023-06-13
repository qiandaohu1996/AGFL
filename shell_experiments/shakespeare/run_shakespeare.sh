source D:/AGFL-main/shell_experiments/run.sh


# DATA=("emnist" emnist_compon4ent "femnist" "cifar10" "cifar100")
# DATA=("shakespeare" )
dataset="shakespeare"
 
algo="FedAvg"

run $dataset  

run $dataset  --locally_tune_clients  

algo="local"

run $dataset  

algo="clustered"

run $dataset   

algo="APFL"

run $dataset  --alpha 0.5  --adaptive_alpha


algo="FedProx"
  
run $dataset   --mu '0.1'

algo="pFedMe"

run $dataset   --mu '0.1'

inner_dir="_mu_0.5"

algo="FedProx"

run $dataset   --mu '0.5'
 
algo="pFedMe"

run $dataset   --mu '0.5'


algo="FedEM"

run $dataset  --sampling_rate '0.5'
