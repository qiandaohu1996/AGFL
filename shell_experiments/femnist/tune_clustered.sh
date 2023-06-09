cd ../../


echo "Run clustered lr=0.003"
python run_experiment.py femnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --validation  \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/clustered_lr_0.003

echo "Run clustered lr=0.01"
python run_experiment.py femnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --validation \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/clustered_lr_0.01

echo "Run clustered lr=0.03"
python run_experiment.py femnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --validation  \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/clustered_lr_0.03

echo "Run clustered lr=0.1"
python run_experiment.py femnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 \
 --lr_scheduler multi_step --log_freq 2 --device cuda --optimizer sgd --validation \
 --seed 1234 --verbose 1 --logs_root logs/femnist_validation/clustered_lr_0.1