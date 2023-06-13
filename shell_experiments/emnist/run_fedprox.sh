echo "Run FedProx"

mkdir -p logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_0.25/

python run_experiment.py emnist_pathologic_cl20 FedProx --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --mu 1.0 --lr_scheduler cosine_annealing --log_freq 2 --optimizer prox_sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/FedProx/batch/FedProx_lr_0.1_mu_0.25 > logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_0.25/emnist_pathologic_cl20_FedProx_lr_0.1_mu_0.25.log

mkdir -p logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_0.5 

python run_experiment.py emnist_pathologic_cl20 FedProx --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --mu 0.5 --lr_scheduler cosine_annealing --log_freq 2 --optimizer prox_sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_0.5 > logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_0.5/emnist_pathologic_cl20_batch_FedProx_lr_0.1_mu_0.5.log

mkdir -p logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_1

python run_experiment.py emnist_pathologic_cl20 FedProx --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --mu 1 --lr_scheduler cosine_annealing --log_freq 2 --optimizer prox_sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_1 > logs/emnist_pathologic_cl20/batch/FedProx/FedProx_lr_0.1_mu_1/emnist_pathologic_cl20_batch_FedProx_lr_0.1_mu_1.log

