adam

echo "Run FuzzyFL lr=0.003"
python run_experiment.py femnist FuzzyFL --n_learners 1 --pre_roumds 10  --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/lr_0.003

echo "Run FuzzyFL lr=0.01"
python run_experiment.py femnist FuzzyFL --n_learners 1 --pre_roumds 10  --n_rounds 200 --bz 128 --lr 0.01  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd
 --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/lr_0.01

echo "Run FuzzyFL lr=0.05"
python run_experiment.py femnist FuzzyFL --n_learners 1  --n_rounds 200 --bz 128 --lr 0.05 --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/lr_0.05 >> femnist/FuzzyFL/lr_0.05.log

echo "Run FuzzyFL lr=0.05"
python run_experiment.py femnist FuzzyFL --n_learners 1 --pre_rounds 10  --n_rounds 200 --bz 128 --lr 0.05 --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/lr_0.05 >> femnist_FuzzyFL_pre10_lr_0.05.log

echo "Run FuzzyFLlr=0.1"

python run_experiment.py femnist FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch   --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/pre25_lr_0.1  >> femnist_FuzzyFL_pre25_lr_0.1.log

python run_experiment.py femnist FuzzyFL --n_learners 1 --pre_rounds 0 --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch   --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/pre0_lr_0.1  >> femnist_FuzzyFL_pre0_lr_0.1.log

python run_experiment.py femnist FuzzyFL --n_learners 1 --pre_rounds 50  --n_rounds 200 --bz 128 --lr 0.075  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/femnist/FuzzyFL/pre50_lr_0.01 >> logs_femnist_FuzzyFL_lr_0.01