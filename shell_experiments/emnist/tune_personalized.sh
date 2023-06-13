cd ../../


echo "Run Personalized (Richtarek's Formulation), mu=0.01"
python run_experiment.py emnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.01 \
 --lr_scheduler constant --log_freq 20 --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/emnist/pFedMe/mu_0.01

echo "Run Personalized (Richtarek's Formulation), mu=0.1"
python run_experiment.py emnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.1 \
 --lr_scheduler constant --log_freq 20 --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/emnist/pFedMe/mu_0.1

echo "Run Personalized (Richtarek's Formulation), mu=1"
python run_experiment.py emnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 1.0 \
 --lr_scheduler constant --log_freq 20 --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/emnist/pFedMe/mu_1

echo "Run Personalized (Richtarek's Formulation), mu=10"
python run_experiment.py emnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 10.0 \
 --lr_scheduler constant --log_freq 20 --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/emnist/pFedMe/mu_10

echo "Run Personalized (Richtarek's Formulation), mu=100"
python run_experiment.py emnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 100.0 \
 --lr_scheduler constant --log_freq 20 --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/emnist/pFedMe/mu_100


LOGS_BASE_DIR="logs/emnist_pathologic_cl20/batch/pFedMe"



LOGS_DIR="$LOGS_BASE_DIR/pFedMe_mu_1"
mkdir -p "$LOGS_DIR"

python run_experiment.py emnist_pathologic_cl20 pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 1.0 --lr_scheduler constant --log_freq 2 --optimizer prox_sgd --minibatch --seed 1234 --verbose 1 --logs_root "$LOGS_DIR" > "$LOGS_DIR/emnist_pathologic_cl20_batch_pFedMe_mu_1.log"

LOGS_DIR="$LOGS_BASE_DIR/pFedMe_mu_1"
mkdir -p "$LOGS_DIR"

python run_experiment.py emnist_pathologic_cl20 pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 1.0 --lr_scheduler constant --log_freq 2 --optimizer prox_sgd --minibatch --seed 1234 --verbose 1 --logs_root "$LOGS_DIR" > "$LOGS_DIR/emnist_pathologic_cl20_batch_pFedMe_mu_1.log"