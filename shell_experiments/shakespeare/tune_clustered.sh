cd ../../


echo "Run clustered lr=0.003"
python run_experiment.py shakespeare clustered --n_learners 1 --n_rounds 100 --bz 128 --lr 0.003 \
 --lr_scheduler constant --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/shakespeare_validation/clustered_lr_0.003

echo "Run clustered lr=0.01"
python run_experiment.py shakespeare clustered --n_learners 1 --n_rounds 100 --bz 128 --lr 0.01 \
 --lr_scheduler constant --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/shakespeare_validation/clustered_lr_0.01

echo "Run clustered lr=0.03"
python run_experiment.py shakespeare clustered --n_learners 1 --n_rounds 100 --bz 128 --lr 0.03 \
 --lr_scheduler constant --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/shakespeare_validation/clustered_lr_0.03

echo "Run clustered lr=0.1"
python run_experiment.py shakespeare clustered --n_learners 1 --n_rounds 100 --bz 128 --lr 0.1 \
 --lr_scheduler constant --log_freq 2 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/shakespeare_validation/clustered_lr_0.1
