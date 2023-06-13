echo "no adaptive"

echo "Run AGFL lr=0.1 alpha0"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0. --local_step 5 --bz 128 --lr 0.1   --log_freq 2 --lr_scheduler multi_step --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0_fixed_step5

echo "Run AGFL lr=0.1 alpha0.25"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.25 --local_step 5 --bz 128 --lr 0.1   --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.25_fixed_step5

echo "Run AGFL lr=0.1 alpha0.5"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2--alpha 0.5 --local_step 3 --bz 128 --lr 0.1   --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_fixed_step5

echo "Run AGFL lr=0.1 alpha0.75"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.75 --local_step 5 --bz 128 --lr 0.1    --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.75_fixed_step5

echo "Run AGFL lr=0.1 alpha1"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 1 --local_step 5 --bz 128 --lr 0.1    --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha1_fixed_step5

# echo "Run AGFL lr=0.001 alpha0.3 adaptive" 
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.3 --adaptive_alpha --local_step 5 --bz 128 --lr 0.001   --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.01_alpha0.3_adapt

# echo "Run AGFL lr=0.001 alpha0.3 adaptive 1batch"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.3 --adaptive_alpha --local_step 5 --bz 128 --lr 0.001 --minibatch  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.01_alpha0.3_adapt_1batch

echo "step5"
echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.1  --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt_step5  >> 1.log

echo "Run AGFL lr=0.12 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.12   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.12_alpha0.5_adapt_step5

echo "Run AGFL lr=0.08 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.08   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.08_alpha0.5_adapt_step5

echo "Run AGFL lr=0.1 alpha 0.5 adaptive bz256"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 256 --lr 0.1   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt_step5_bz256

echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 2 --bz 128 --lr 0.1   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt_step2

echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 8 --bz 128 --lr 0.1 --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt_step8

echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 10 --bz 128 --lr 0.1  --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.001_alpha0.5_adapt_step10

# echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.1   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.001_alpha0.5_adapt
# echo "Run AGFL lr=0.001 alpha 0.5 adaptive 1batch"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.001   --log_freq 2 --minibatch --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.01_alpha0.5_adapt_1batch

# echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.1   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt

# echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.1   --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt

# echo "Run AGFL lr=0.12 alpha0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.12  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt

# echo "Run AGFL lr=0.01 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha  --local_step 5 --bz 128 --lr 0.01     --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.01_alpha0.5_adapt

# echo "Run AGFL lr=0.01 alpha 0.5 adaptive 1batch"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.01   --minibatch --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.01_alpha0.5_adapt_1batch

# echo "Run AGFL lr=0.1 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 5 --bz 128 --lr 0.1   --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt

# # echo "alpha 0.8 "
# echo "Run AGFL lr=0.3 alpha 0.5 adaptive"
# python run_experiment.py emnist AGFL --pre_rounds 50 --n_rounds 200 --n_learners 2 --alpha 0.5 --adaptive_alpha  --local_step 5 --bz 128 --lr 0.3    --log_freq 2 --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/agfl/saga/lr_0.03_alpha0.5_adapt

 
echo "Run AGFL lr=0.1 alpha0.5"
# python run_experiment.py emnist AGFL --pre_rounds 1 --n_rounds 2 --n_learners 2 --alpha 0.5 --local_step 4 --bz 128 --lr 0.1   --optimizer sgd --seed 1234 --verbose 1 --logs_root 112 >> 1.log

python run_experiment.py emnist2 AGFL --pre_rounds 2 --n_rounds 4 --n_learners 2 --alpha 0.5 --adaptive_alpha --local_step 4 --bz 128 --lr 0.05   --log_freq 2 --optimizer sgd --seed 1234 --verbose 2 --logs_root logs/emnist/agfl/saga/lr_0.1_alpha0.5_adapt_step5    >> 6.log
