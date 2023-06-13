#!/usr/bin/env bash
# shellcheck disable=SC2034

declare -g DATA=()
declare -g algos=()

# algorithm parameters
declare -g algo=""
declare -g lr=0.1
declare -g bz=64
declare -g local_steps=4

declare -g n_rounds=200
declare -g log_freq=2

declare -g sampling_rates=("0.5")

# for fuzzyFL
declare -g pre_rounds_list=("25")
declare -g fuzzy_m_schedulers=("constant")
declare -g fuzzy_m_list=("1.75")
declare -g trans_list=("0.75")
declare -g fuzzy_m_momentums=("0.8")
 

# for l2sgd
declare -g comm_probs=("0.2")  
declare -g mus=("0.1") 

set_inner_dir() {
    local param="$1"
    declare -A paramToTemplate
    paramToTemplate=( ["pre_rounds"]="_pre_%s" 
                      ["fuzzy_m"]="_m_%s" 
                      ["sampling_rate"]="_samp_%s"
                      ["locally_tune_clients"]="_tune"
                      ["adaptive"]="adapt"
                      ["fuzzy_m_scheduler"]="_sch_%s"
                      ["fuzzy_m_momentum"]="_mt_%s"
                      ["n_clusters"]="_cluster_%s"
                       )
    
    if [[ -v "parameters[$param]" ]]; then
        if [[ "${parameters[$param]}" != "true" ]]; then
            local template=${paramToTemplate[$param]}
            if [[ -z "$template" ]]; then
                template="_${param}_%s" # default template
            fi
            # shellcheck disable=SC2059
            inner_dir+=$(printf "$template"   "${parameters[$param]}")

        else
            inner_dir+="_${param}"
        fi
    fi
}



run_base() {
    local dataset="$1"
    shift
    local log_type="gd"
    local n_learners=1
    local optimizer="sgd"
    local extra_args_str=("$@")  
    declare -A parameters

    echo "${extra_args_str[@]}"

    if [[ $dataset == synthetic* ]]; then
        extra_args_str+=("--input_dimension" "150" "--output_dimension" "2")
    fi   
    while (( "$#" )); do
        if [[ $1 == --* ]]; then
            local key=${1#--}
            shift
            if [[ $1 != --* && $1 != "" ]]; then
                parameters[$key]=$1
                shift
            else
                parameters[$key]=true
            fi
        else
            shift
        fi
    done
    parameters["sampling_rate"]=$sampling_rate

    if [[ -v parameters["fuzzy_m_scheduler"] ]]; then
        case ${parameters["fuzzy_m_scheduler"]} in
            "multi_step")
                parameters["fuzzy_m_scheduler"]="2step"
                ;;
            "cosine_annealing")
                parameters["fuzzy_m_scheduler"]="cosine"
                ;;
        esac
    fi

    local samp_dir="" 
    local inner_dir=""

    case $algo in
        "FedAvg")
            set_inner_dir "locally_tune_clients" 
            ;;
        "FuzzyFL")
            set_inner_dir "pre_rounds"
            set_inner_dir "fuzzy_m"
            set_inner_dir "trans"
            set_inner_dir "fuzzy_m_scheduler"
            set_inner_dir "fuzzy_m_momentum"
            set_inner_dir "n_clusters"
            samp_dir="/samp$sampling_rate"
            ;;
        "APFL")
            n_learners=2
            set_inner_dir "alpha"
            set_inner_dir "adaptive"
            ;;
        "clustered")
            sampling_rate=1
            ;;
        "AGFL")
            n_learners=4
            set_inner_dir "alpha"
            set_inner_dir "adaptive"
            set_inner_dir "pre_rounds"
            ;;
        "FedEM")
            n_learners=3
            set_inner_dir "m"
            ;;
        "FedProx")
            set_inner_dir "mu"
            optimizer="prox_sgd"
            ;;
        "pFedMe")
            set_inner_dir "mu"
            optimizer="prox_sgd"
            ;;
        "L2SGD")
            set_inner_dir "mu"
            set_inner_dir "comm_prob"
            optimizer="prox_sgd"
            ;;
        *)
            ;;
    esac
    
    set_inner_dir "sampling_rate"
    if [[ ${parameters[minibatch]} ]]; then
        log_type="batch"
    fi 
     
    local log_dir="logs/$dataset/$log_type/${algo}${samp_dir}/${algo}_lr_${lr}${inner_dir}"
    local out_file="${dataset}_${algo}_lr_${lr}${inner_dir}.log"

    check_dir "$log_dir"

    echo "$out_file"

    python run_experiment.py "$dataset" "$algo" --n_rounds $n_rounds --n_learners $n_learners --sampling_rate "$sampling_rate"  --lr $lr --lr_scheduler cosine_annealing --optimizer $optimizer --bz $bz --local_steps $local_steps  --log_freq $log_freq --seed 1234 --verbose 1 "${extra_args_str[@]}" --logs_root "$log_dir"  > "$log_dir/$out_file"
}


run() {
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
            run_base "$dataset" "$@" --minibatch 
        done
    done
}

run_fuzzy() {
algo="FuzzyFL"

for pre_rounds in "${pre_rounds_list[@]}"; do
for m in "${fuzzy_m_list[@]}"; do
    for trans in "${trans_list[@]}"; do
    for fuzzy_m_scheduler in "${fuzzy_m_schedulers[@]}"; do
        for fuzzy_m_momentum in "${fuzzy_m_momentums[@]}"; do
        run --pre_rounds "$pre_rounds" --fuzzy_m "$m" --trans "$trans"  --fuzzy_m_scheduler "${fuzzy_m_scheduler}" --fuzzy_m_momentum "${fuzzy_m_momentum}"
        done
    done
    done
done
done
}

run_avg(){
    algo="FedAvg" 
    run
}

run_avg_adap(){
    algo="FedAvg" 
    run --locally_tune_clients 
}

run_local(){
    algo="local" 
    run  
}
run_em(){
    algo="FedEM" 
    run
}

run_clustered(){
    algo="clustered" 
    run
}

run_avgem(){
    algos=("FedAvg" "FedEM")
    for algo in "${algos[@]}"; do
        run
    done
}

run_l2gd(){
algo="L2SGD"
for comm_prob in "${comm_probs[@]}"; do
    for mu in "${mus[@]}"; do
        run  --comm_prob "$comm_prob"  --mu  "$mu" 
    done
done
}

run_pfedme(){
    algo="pFedMe" 
    for mu in "${mus[@]}"; do
        run  --mu  "$mu" 
    done
}

run_prox(){
    algo="FedProx"
    for mu in "${mus[@]}"; do
        run  --mu  "$mu" 
    done
}



check_dir() {
  local dir_path="$1"

  if [ -d "$dir_path" ]; then
    echo -e "Directory $dir_path exists. \nIt will be deleted in 15 seconds unless you cancel the operation. \nDo you want to remove it? (Yy/n)"
    
    count=5
    while [ $count -gt 0 ]; do
      echo -ne "\rTime remaining: $count s..."
      if read -r -t 1 user_input; then
        if [[ "$user_input" =~ ^(y|Y)$ ]]; then
          rm -rf "$dir_path"
          break
        elif [ "$user_input" = "n" ]; then
          echo -e "\nDirectory $dir_path not removed. Now will rename $dir_path"
          
          suffix=1
          while [ -d "${dir_path}_$suffix" ]; do
            suffix=$((suffix+1))
          done
          new_dir_path="${dir_path}_$suffix"
          
          mv "$dir_path" "$new_dir_path"
          echo "Directory $dir_path renamed to $new_dir_path"
          
          mkdir -p "$dir_path"
          echo "New directory $dir_path created."
          return 0   
        fi
      fi
      count=$((count-1))
    done

      echo -e "\nDirectory $dir_path removed due to no response."
      rm -rf "$dir_path"
      
      echo "Now creating a new directory $dir_path"
      mkdir -p "$dir_path"
      echo "$dir_path created successfully"
  else 
    echo "Directory $dir_path does not exist, now creating."
    mkdir -p "$dir_path"
    echo "$dir_path created successfully"
  fi
}


run_gd() {
    for dataset in "${DATA[@]}"; do
        for sampling_rate in "${sampling_rates[@]}"; do
            run_base "$dataset" "$@" 
        done
    done
}

show_dict(){
local dict="$1"
for key in "${!dict[@]}"; do
        value="${dict[$key]}"
        echo "$key: $value"
done
}
modify_dict_key() {
    local dict_var="$1"
    local old_key="$2"
    local new_key="$3"

    eval "${dict_var}[\"$new_key\"]=\"\${${dict_var}[\"$old_key\"]}\""

    unset "${dict_var}[$old_key]"
}