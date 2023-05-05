cd ../../
echo "Run AGFL lr=0.003"
python run_experiment.py emnist APFL  --n_rounds 10 --bz 16 --lr 0.003  --lr_scheduler multi_step --log_freq 10  --optimizer sgd --validation   --seed 1234 --verbose 1 --logs_root logs/emnist_validation/apfl_lr_0.003
