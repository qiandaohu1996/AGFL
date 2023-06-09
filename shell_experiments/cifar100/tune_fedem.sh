cd ../../


echo "Run FedEM lr=0.003"
python run_experiment.py cifar100 FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.003 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedEM_lr_0.003

echo "Run FedEM lr=0.01"
python run_experiment.py cifar100 FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedEM_lr_0.01

echo "Run FedEM lr=0.03"
python run_experiment.py cifar100 FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedEM_lr_0.03

echo "Run FedEM lr=0.1"
python run_experiment.py cifar100 FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.1 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedEM_lr_0.1
