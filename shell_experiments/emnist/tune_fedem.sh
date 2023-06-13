cd ../../

echo "minibatch"

echo "Run FedEM lr=0.003"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.003  --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/emnist_FedEM_lr_0.003

echo "Run FedEM lr=0.01"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.01  --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/emnist_FedEM_lr_0.01

echo "Run FedEM lr=0.03"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03  --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/emnist_FedEM_lr_0.03_batch >> logs/emnist/emnist_FedEM_lr_0.03_batch/emnist_FedEM_lr_0.03_batch.log

echo "Run FedEM lr=0.1"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.1  --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/emnist_FedEM_lr_0.1



echo "Run FedEM lr=0.003"


python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.003  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist/FedEM_lr_0.003

echo "Run FedEM lr=0.01"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.01  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist/FedEM_lr_0.01

echo "Run FedEM lr=0.03"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist/FedEM_lr_0.03

echo "Run FedEM lr=0.1"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.1  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist/FedEM_lr_0.1


echo "emnist_pathologic_validation"


echo "Run FedEM lr=0.003"


python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.003  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FedEM_lr_0.003

echo "Run FedEM lr=0.01"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.01  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FedEM_lr_0.01

echo "Run FedEM lr=0.03"
python run_experiment.py emnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FedEM_lr_0.03

echo "Run FedEM lr=0.1"
python run_experiment.py emnist_pathologic_cl10 FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.1  --log_freq 2 --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl10_validation/FedEM/FedEM_lr_0.1 > logs/emnist_pathologic_cl10_validation/FedEM/FedEM_lr_0.1/emnist_pathologic_cl10_validation_FedEM_lr_0.1.log


echo "Run FedEM lr=0.1 test emnist2"

 python run_experiment.py femnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --minibatch --lr 0.1 --log_freq 2 --optimizer sgd --validation --seed 1234 --verbose 1 --logs_root logs/emnist2/FedEM_lr_0.1