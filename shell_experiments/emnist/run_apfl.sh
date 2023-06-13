 
echo "Run APFL lr=0.003 alpha_0.3 adaptive_alpha"
python run_experiment.py emnist APFL --n_rounds 200 --alpha 0.3 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.003  --lr_scheduler linear --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.003_alpha_0.3_adapt


echo "Run APFL lr=0.01 alpha_0.3 adaptive_alpha"
python run_experiment.py emnist APFL --n_rounds 200 --alpha 0.3 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.01  --lr_scheduler linear --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.01_alpha_0.3_adapt_local_lr_0.005

echo "Run APFL lr=0.03 alpha_0.5 adaptive_alpha"
python run_experiment.py emnist APFL --n_rounds 200 --alpha 0.5  --adaptive_alpha --n_learners 2 --bz 128 --lr 0.03  --lr_scheduler linear --log_freq 2  --optimizer sgd --minibatch  --seed 1234 --verbose 1 --logs_root logs/emnist/apfl/apfl_lr_0.03_alpha_0.5_adapt_batch > logs/emnist/apfl/apfl_lr_0.03_alpha_0.5_adapt_batch/apfl_lr_0.03_alpha_0.5_adapt_batch.log

echo "Run APFL lr=0.005 alpha 0.5 adaptive_alpha"
python run_experiment.py emnist APFL --n_rounds 200 --alpha 0.5 --adaptive_alpha  --n_learners 2 --bz 128 --lr 0.005  --lr_scheduler linear --log_freq 2  --optimizer sgd  --minibatch  --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.005_alpha0.5_adapt_lr_0.005

echo "Run APFL lr=0.03 alpha 0.8 adaptive_alpha"
python run_experiment.py emnist APFL --n_rounds 200 --alpha 0.8 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.03  --lr_scheduler linear --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.03_alpha_0.8_adapt

echo "Run APFL lr=0.1 alpha 0.5 adaptive_alpha"
python run_experiment.py emnist APFL --pre_rounds 20 --n_rounds 50 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler linear --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.03_alpha_0.8_adapt

echo "Run APFL lr=0.1 alpha 0.5 adaptive_alpha"
python run_experiment.py emnist APFL --pre_rounds 20 --n_rounds 50 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler linear --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.03_alpha_0.8_adapt

echo "Run APFL lr=0.1 alpha 0.8 adaptive_alpha"
python run_experiment.py emnist APFL --n_rounds 200 --alpha 0.8 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler linear --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/apfl_lr_0.03_alpha_0.8_adapt


