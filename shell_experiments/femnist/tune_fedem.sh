cd ../../


echo "Run FedEM lr=0.003"
python run_experiment.py femnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.003 \
 --log_freq 2  --optimizer sgd --validation  \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/FedEM_lr_0.003

echo "Run FedEM lr=0.01"
python run_experiment.py femnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.01 \
 --log_freq 2  --optimizer sgd --validation \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/FedEM_lr_0.01

echo "Run FedEM lr=0.03"
python run_experiment.py femnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03 \
 --log_freq 2  --optimizer sgd --validation  \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/FedEM_lr_0.03

echo "Run FedEM lr=0.1"
python run_experiment.py femnist FedEM --n_learners 3 --sampling_rate 0.5 --n_rounds 200 --bz 128 --lr 0.1 --log_freq 2  --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/femnist_validation/FedEM_lr_0.1 >>FedEM_lr_0.1.log