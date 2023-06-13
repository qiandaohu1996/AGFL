cd ../../


echo "Run clustered lr=0.003"
python run_experiment.py emnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003 --log_freq 2 --optimizer  sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/clustered/clustered_lr_0.003

echo "Run clustered lr=0.01"
python run_experiment.py emnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --log_freq 2 --optimizer  sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/clustered/clustered_lr_0.01

echo "Run clustered lr=0.03"
python run_experiment.py emnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --log_freq 2 --optimizer  sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/clustered/clustered_lr_0.03_batch > logs/emnist/clustered/clustered_lr_0.03_batch/clustered_lr_0.03_batch.log

echo "Run clustered lr=0.1"
python run_experiment.py emnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --log_freq 2 --optimizer  sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/clustered/clustered_lr_0.1


echo "Run clustered lr=0.003"

echo "Run clustered lr=0.003"
python run_experiment.py emnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003 --log_freq 2 --optimizer  sgd --validation --seed 1234 --verbose 1 --logs_root logs/emnist/clustered/clustered_lr_0.003

echo "Run clustered lr=0.01"
python run_experiment.py emnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --log_freq 2 --optimizer  sgd --validation --seed 1234 --verbose 1 --logs_root logs/emnist/clustered/clustered_lr_0.01

echo "Run clustered lr=0.03"
python run_experiment.py emnist_pathologic_cl10 clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --log_freq 2 --optimizer  sgd  --validation --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/clustered/clustered_lr_0.03 > logs/emnist_pathologic_cl10_validation/clustered/clustered_lr_0.03/emnist_pathologic_cl10_validation_clustered_lr_0.03.log

echo "Run clustered lr=0.1"
python run_experiment.py emnist_pathologic_cl10 clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --log_freq 2 --optimizer  sgd  --validation --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/clustered/clustered_lr_0.1 > logs/emnist_pathologic_cl10_validation/clustered/clustered_lr_0.1/emnist_pathologic_cl10_validation_clustered_lr_0.1.log