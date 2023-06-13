echo "no adaptive"

echo "Run AGFL lr=0.1 alpha0"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0. --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0

echo "Run AGFL lr=0.1 alpha0.25"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.25 --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0.25

echo "Run AGFL lr=0.1 alpha0.5"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0.5

echo "Run AGFL lr=0.1 alpha0.75"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.75 --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd  --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0.75

echo "Run AGFL lr=0.1 alpha1"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 1 --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd  --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0.75

# echo "Run AGFL lr=0.001 alpha0.3 adaptive" 
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.3 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.001  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.01_alpha0.3_adapt

# echo "Run AGFL lr=0.001 alpha0.3 adaptive 1batch"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.3 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.001 --minibatch --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.01_alpha0.3_adapt_1batch


echo "Run AGFL lr=0.001 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.001 --lr_scheduler multi_step  --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.001_alpha0.5_adapt

echo "Run AGFL lr=0.001 alpha 0.5 adaptive 1batch"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.001 --lr_scheduler multi_step  --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.01_alpha0.5_adapt_1batch

echo "Run AGFL lr=0.003 alpha0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.003 --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.003_alpha0.5_adapt

echo "Run AGFL lr=0.003 alpha0.5 adaptive 1batch"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.003 --lr_scheduler multi_step --log_freq 2  --optimizer sgd --minibatch  --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.003_alpha0.5_adapt_1batch

# echo "Run AGFL lr=0.01 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha  --n_learners 2 --bz 128 --lr 0.01  --lr_scheduler multi_step  --log_freq 2 --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.01_alpha0.5_adapt

# echo "Run AGFL lr=0.01 alpha 0.5 adaptive 1batch"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.01  --lr_scheduler multi_step --log_freq 2  --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.01_alpha0.5_adapt_1batch

echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd   --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0.5_adapt

echo "Run AGFL lr=0.1 alpha 0.5 adaptive 1batch"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha --n_learners 2 --bz 128 --lr 0.1  --lr_scheduler multi_step --log_freq 2  --optimizer sgd  --minibatch  --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.1_alpha0.5_adapt_1batch

# echo "alpha 0.8 "
echo "Run AGFL lr=0.3 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5 --adaptive_alpha  --n_learners 2 --bz 128 --lr 0.3  --lr_scheduler multi_step  --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.03_alpha0.5_adapt

echo "Run AGFL lr=0.3 alpha0.5 adaptive 1batch"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --alpha 0.5  --adaptive_alpha --n_learners 2 --bz 128 --lr 0.3  --lr_scheduler multi_step  --log_freq 2 --optimizer sgd --minibatch --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/lr_0.03_alpha0.5_adapt_1batch

# # test
# python run_experiment.py emnist AGFL --pre_rounds 5 --n_rounds 10 --alpha 0.5   --n_learners 2  --bz 128 --minibatch --lr 0.2  --lr_scheduler multi_step  --log_freq 1 --optimizer sgd   --seed 1234 --verbose 1 --logs_root ./111



python run_experiment.py emnist AGFL --pre_rounds 1 --n_rounds 20 --n_learners 2 --alpha 0.5 --local_step 5 --bz 128 --lr 0.1   --log_freq 2 --lr_scheduler multi_step --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root ./111 >>2.log

