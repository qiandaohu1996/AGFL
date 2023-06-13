#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source D:/AGFL-main/shell_experiments/run.sh

# DATA=("emnist_pathologic_cl20" "emnist_component4" "femnist" "cifar10" "cifar100")
# DATA=("cifar10_s0.2" "emnist50" "emnist50_c4")
# DATA=("cifar10_s0.2" )
DATA=("cifar10")
 
sampling_rates=("0.2")

run_em
 