 
echo "Run L2SGD lr=0.003 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --comm_prob 0.1 --bz 128 --lr 0.003 --mu 0.1  --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.003_mu0.1


echo "Run L2SGD lr=0.01 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --comm_prob 0.1 --bz 128 --lr 0.01 --mu 0.1  --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.01_mu0.1

echo "Run L2SGD lr=0.03 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --comm_prob 0.1 --bz 128 --lr 0.03 --mu 0.1  --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03_mu0.1

echo "Run L2SGD lr=0.1 mu=0.1"
python run_experiment.py emnist_pathologic L2SGD --n_rounds 200  --n_learners 1 --comm_prob 0.2  --bz 128 --lr 0.1 --mu 0.1  --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/L2GD/l2sgd_lr_0.1_mu_0.1_com_0.2 > logs/emnist_pathologic_validation/L2GD/l2sgd_lr_0.1_mu_0.1_com_0.2/emnist_pathologic_validation_l2sgd_lr_0.1_mu_0.1_com_0.2.log


echo "Run L2SGD lr=0.03"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --comm_prob 0.2 --bz 128 --lr 0.03   --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03

echo "Run L2SGD lr=0.1"
python run_experiment.py emnist L2SGD --n_rounds 50 --n_learners 1 --comm_prob 0.2 --bz 128 --lr 0.1   --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03

echo "Run L2SGD lr=0.1"
python run_experiment.py emnist L2SGD --n_rounds 50 --n_learners 1 --bz 128 --lr 0.1   --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.1

echo "Run L2SGD lr=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --bz 128 --lr 0.1   --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03
