sgd

echo "Run FuzzyFL lr=0.003"
python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 50  --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/FuzzyFL/lr_0.003

echo "Run FuzzyFL lr=0.01"
python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 50  --n_rounds 200 --bz 128 --lr 0.01 \ --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --validation \
 --seed 1234 --verbose 1 --logs_root logs/emnist/FuzzyFL/lr_0.01


echo "Run FuzzyFL lr=0.03"
python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 20  --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler cosine_annealing --log_freq 1  --optimizer sgd --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_validation/FuzzyFL/lr_0.03 > emnist2_validation_FuzzyFL_lr_0.03.log

echo "Run FuzzyFLlr=0.1"
python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --minibatch --validation  --seed 1234 --verbose 1 --logs_root logs/emnist/FuzzyFL/lr_0.1_batch > emnist_validation_FuzzyFL_lr_0.1.log


python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 1  --n_rounds 200 --bz 128 --lr 0.075  --lr_scheduler cosine_annealing --log_freq 1  --optimizer sgd --validation  --seed 1234 --verbose 1  


echo "emnist_pathologic"

echo "Run FuzzyFL lr=0.003"
python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 50  --n_rounds 200 --bz 128 --lr 0.003  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --seed 1234 --verbose 1 --logs_root logs/emnist/FuzzyFL/lr_0.003

echo "Run FuzzyFL lr=0.01"
python run_experiment.py emnist_pathologic FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FuzzyFL/lr_0.01_m_2 > logs/emnist_pathologic_validation/FuzzyFL/lr_0.01_m_2/emnist_pathologic_validation_FuzzyFL_m2_lr_0.01.log


echo "Run FuzzyFL lr=0.03"
python run_experiment.py emnist FuzzyFL --n_learners 1 --pre_rounds 20  --n_rounds 200 --bz 128 --lr 0.03  --lr_scheduler cosine_annealing --log_freq 1  --optimizer sgd --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_validation/FuzzyFL/lr_0.03_m_2 > emnist2_validation_FuzzyFL_lr_0.03_m_2.log

echo "Run FuzzyFL  lr_0.1 pre_25 m_2"

python run_experiment.py emnist_pathologic FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1 --fuzzy_m 2  --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FuzzyFL/FuzzyFL_lr_0.1_pre_25_m_2 > logs/emnist_pathologic_validation/FuzzyFL/FuzzyFL_lr_0.1_pre_25_m_2/emnist_pathologic_validation_FuzzyFL_lr_0.1_pre_25_m_2.log


echo "Run FuzzyFL  lr_0.1 pre_25 m_2"

python run_experiment.py emnist_pathologic FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1 --fuzzy_m 1.5  --lr_scheduler linear --log_freq 2  --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FuzzyFL/FuzzyFL_lr_0.1_pre_25_m_1.5 > logs/emnist_pathologic_validation/FuzzyFL/FuzzyFL_lr_0.1_pre_25_m_1.5/emnist_pathologic_validation_FuzzyFL_lr_0.1_pre_25_m_1.5.log


python run_experiment.py emnist_pathologic FuzzyFL --n_learners 1 --pre_rounds 25  --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler cosine_annealing --log_freq 2  --optimizer sgd --validation  --seed 1234 --verbose 1 --logs_root logs/emnist_pathologic_validation/FuzzyFL/lr_0.1_pre_50_m_2 > logs/emnist_pathologic_validation/FuzzyFL/lr_0.1_pre_25_m_2/emnist_pathologic_validation_FuzzyFL_lr_0.1_pre_25_m_2.log

