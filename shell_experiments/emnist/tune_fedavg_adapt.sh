cd ../../


echo "Run FedAvg lr=0.003"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --minibatch    --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.003 > logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.003/emnist_FedAvg_adapt_lr_0.003.log

echo "Run FedAvg lr=0.01"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --minibatch   --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.01 > logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.01/emnist_FedAvg_adapt_lr_0.01.log

echo "Run FedAvg lr=0.03"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --minibatch    --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.03 > logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.03/emnist_FedAvg_adapt_lr_0.03.log

echo "Run FedAvglr=0.1"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --minibatch   --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_adapt/FedAvg_adapt_lr_0.1




echo "Run FedAvg lr=0.003"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_validation/FedAvg_adapt_lr_0.003

echo "Run FedAvg lr=0.01"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_validation/FedAvg_adapt_lr_0.01

echo "Run FedAvg lr=0.03"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 2  --optimizer sgd --validation  \
 --seed 1234 --verbose 1 --logs_root logs/emnist_validation/FedAvg_adapt_lr_0.03

echo "Run FedAvglr=0.1"
python run_experiment.py emnist FedAvg --locally_tune_clients --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 \
 --lr_scheduler multi_step --log_freq 2  --optimizer sgd --validation \
 --seed 1234 --verbose 1 --logs_root logs/emnist_validation/FedAvg_adapt_lr_0.1