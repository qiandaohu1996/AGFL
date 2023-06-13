#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091
source D:/AGFL-main/shell_experiments/run.sh
sampling_rates=("0.5")

# DATA=("emnist" "emnist50_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25")

# DATA=("emnist_pathologic_cl20" "emnist_component4")
 
DATA=("femnist")

# DATA=("cifar10")

pre_rounds_list=("25")
# n_clusters=5
fuzzy_m_list=("1.75")
trans_list=( "0.75" )
fuzzy_m_momentums=("0.8")
fuzzy_m_schedulers=("cosine_annealing")

run_fuzzy