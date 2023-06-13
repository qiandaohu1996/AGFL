#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source D:/AGFL-main/shell_experiments/run.sh

# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

DATA=("femnist" )

# sampling_rates=( "0.5")
# pre_rounds_list=("1" "25")
# fuzzy_m_list=( "1.75" "1.6" )
# fuzzy_m_momentums=( "0.8" "0.5"  )
# trans_list=( "0.75")
# fuzzy_m_schedulers=("cosine_annealing" "constant")

# run_fuzzy  

 

sampling_rates=(  "0.5")

# run_avg


mus=("0.1" "1")
run_prox
run_pfedme

comm_prob=("0.5")
run_l2gd
run_clustered

pre_rounds_list=("1")
fuzzy_m_list=( "1.75"  )
fuzzy_m_momentums=( "0.8" )
trans_list=( "0.75")
fuzzy_m_schedulers=("cosine_annealing" "constant")

run_fuzzy

DATA=( "cifar100" )

sampling_rates=( "0.2")

run_avgem

pre_rounds_list=("1" )

fuzzy_m_list=( "2.2"  "2")
trans_list=( "0.75")
fuzzy_m_momentums=( "0.8")
fuzzy_m_schedulers=("cosine_annealing" "constant")

run_fuzzy     

 

# pre_rounds=25
# fuzzy_m_schedulers=("constant")  
# run_fuzzy  

 