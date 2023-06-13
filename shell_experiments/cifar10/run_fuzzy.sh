adam

echo "Run FuzzyFL lr=0.003"
python run_experiment.py cifar10 FuzzyFL --n_learners 1 --pre_roumds 50  --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer adam --seed 1234 --verbose 1 --logs_root logs/cifar10/FuzzyFL/lr_0.003

echo "Run FuzzyFL lr=0.01"
python run_experiment.py cifar10 FuzzyFL --n_learners 1 --pre_roumds 50  --n_rounds 200 --bz 128 --lr 0.01 \ --lr_scheduler cosine_annealing --log_freq 2  --optimizer adam --validation \
 --seed 1234 --verbose 1 --logs_root logs/cifar10/FuzzyFL/lr_0.01

echo "Run FuzzyFL lr=0.03"
python run_experiment.py cifar10 FuzzyFL --n_learners 1 --pre_roumds 50  --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler cosine_annealing --log_freq 2  --optimizer adam --validation  \
 --seed 1234 --verbose 1 --logs_root logs/cifar10/FuzzyFL/lr_0.03

echo "Run FuzzyFLlr=0.1"
python run_experiment.py cifar10 FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer adam --minibatch --validation  --seed 1234 --verbose 1 --logs_root logs/cifar10/FuzzyFL/lr_0.1 


python run_experiment.py cifar10 FuzzyFL --n_learners 1 --pre_rounds 1  --n_rounds 200 --bz 128 --lr 0.08  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sfg  --seed 1234 --verbose 1 --logs_root logs/cifar10/FuzzyFL/lr_0.1 



run_experiment.py cifar100 FuzzyFL --n_learners 1 --pre_rounds 10 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler cosine_annealing --log_freq 2 --optimizer sgd --minibatach --seed 1234 --verbose 1 --logs_root logs/cifar100/FuzzyFL/lr_0.111
