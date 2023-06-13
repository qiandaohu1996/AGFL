cd ../../



echo "Run FedAvg lr=0.003"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch   --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_lr_0.003_batch

echo "Run FedAvg lr=0.01"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 2 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch  --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_lr_0.01_batch

echo "Run FedAvg lr=0.03"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch   --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg/FedAvg_lr_0.03_batch > 'logs/emnist/FedAvg/emnist_FedAvg_lr_0.03_batch.log'

echo "Run FedAvglr=0.1"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch  --seed 1234 --verbose 1 --logs_root logs/emnist/FedAvg_lr_0.1_batch > logs/emnist/FedAvg/emnist_FedAvg_lr_0.03_batch


echo "Run FedAvg validation"


echo "Run FedAvg lr=0.003"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_cl20/FedAvg_lr_0.003

echo "Run FedAvg lr=0.01"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 2 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_cl20/FedAvg_lr_0.01

echo "Run FedAvg lr=0.03"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_cl20/FedAvg_lr_0.03

echo "Run FedAvglr=0.1"
python run_experiment.py emnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_cl20/FedAvg_lr_0.1 >> emnist_cl20_FedAvg_lr_0.1.log


echo "Run FedAvg validation pathologic"


echo "Run FedAvg lr=0.003"
python run_experiment.py emnist_pathologic FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/FedAvg_lr_0.003

echo "Run FedAvg lr=0.01"
python run_experiment.py emnist_pathologic FedAvg --n_learners 1 --n_rounds 2 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/FedAvg_lr_0.003

echo "Run FedAvg lr=0.03"
python run_experiment.py emnist_pathologic FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd  --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/FedAvg_lr_0.003

echo "Run FedAvglr=0.1"
python run_experiment.py emnist_pathologic_cl20 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler  cosine_annealing --minibatch --log_freq 2  --optimizer sgd  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_cl20/batch/FedAvg/FedAvg_lr_0.1 > logs/emnist_pathologic_cl20/batch/FedAvg/FedAvg_lr_0.1/emnist_pathologic_cl20_batch_FedAvg_lr_0.1.log 



