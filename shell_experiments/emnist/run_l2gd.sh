 
echo "Run L2SGD lr=0.003 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --com_prob 0.1 --bz 128 --lr 0.003 --mu 0.1 --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.003_mu0.1


echo "Run L2SGD lr=0.01 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --com_prob 0.1 --bz 128 --lr 0.01 --mu 0.1 --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.01_mu0.1

echo "Run L2SGD lr=0.03 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --com_prob 0.1 --bz 128 --lr 0.03 --mu 0.1 --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03_mu0.1

echo "Run L2SGD lr=0.1 mu=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200  --n_learners 1 --com_prob 0.2  --bz 128 --lr 0.1 --mu 0.1 --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.005_mu0.1

echo "Run L2SGD lr=0.03"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --com_prob 0.2 --bz 128 --lr 0.03  --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03

echo "Run L2SGD lr=0.1"
python run_experiment.py emnist L2SGD --n_rounds 50 --n_learners 1 --com_prob 0.2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03

echo "Run L2SGD lr=0.1"
python run_experiment.py emnist L2SGD --n_rounds 50 --n_learners 1 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03

echo "Run L2SGD lr=0.1"
python run_experiment.py emnist L2SGD --n_rounds 200 --n_learners 1 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 5  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/l2sgd_lr_0.03
