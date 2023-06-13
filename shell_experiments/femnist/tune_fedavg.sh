cd ../../


echo "Run FedAvg lr=0.003"
python run_experiment.py femnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2 --optimizer sgd --minibatch  --seed 1234 --verbose 1 --logs_root logs/femnist/FedAvg/lr_0.003

echo "Run FedAvg lr=0.01"
python run_experiment.py femnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01  --lr_scheduler cosine_annealing --log_freq 2 --optimizer sgd  --minibatch  --seed 1234 --verbose 1 --logs_root logs/femnist/FedAvg/lr_0.01

echo "Run FedAvg lr=0.05"
python run_experiment.py femnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.05  --lr_scheduler cosine_annealing --log_freq 2 --optimizer sgd  --minibatch  --seed 1234 --verbose 1 --logs_root logs/femnist/FedAvg/lr_0.05

echo "Run FedAvg lr=0.1"
python run_experiment.py femnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr_scheduler cosine_annealing --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/femnist/FedAvg/lr_0.1 >> femnist_FedAvg_lr_0.1