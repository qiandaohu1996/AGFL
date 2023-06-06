
declare -g DATA=()
declare -g algo=""
declare -g lr=0.1

set_inner_dir() {
    local param="$1"
    declare -A paramToTemplate
    paramToTemplate=( ["pre_rounds"]="_pre_%s" 
                      ["fuzzy_m"]="_m_%s" 
                      ["locally_tune_clients"]="locally_tune"
                      ["adaptive"]="adapt"
                       )
    
    if [[ -v "parameters[$param]" ]]; then
        if [[ "${parameters[$param]}" != "true" ]]; then
            local template=${paramToTemplate[$param]}
            if [[ -z "$template" ]]; then
                template="_${param}_%s" # default template
            fi
            inner_dir+=`printf "$template" "${parameters[$param]}"`
        else
            inner_dir+="_${param}"
        fi
    fi
}


modify_dict_key() {
    local dict_var="$1"
    local old_key="$2"
    local new_key="$3"

    # 复制原始键值对到新的键名下
    eval "${dict_var}[\"$new_key\"]=\"\${$dict_var[\"$old_key\"]}\""

    # 删除原始键名和对应的值
    unset "$dict_var[$old_key]"
}

run() {
    local dataset="$1"
    shift
    local log_type="gd"
    local n_learners=1
    local optimizer="sgd"
    local extra_args_str="$@"  
    declare -A parameters

    # echo $extra_args_str 

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


    inner_dir=""
    case $algo in
        "FedAvg")
            if [[ -v "parameters[$locally_tune_clients]" ]]; then
                inner_dir+="locally_tune"
            fi
            ;;
        "APFL")
            n_learners=2
            set_inner_dir "alpha"
            set_inner_dir "adaptive"
            ;;
        "AGFL")
            n_learners=4
            set_inner_dir "alpha"
            set_inner_dir "adaptive"
            set_inner_dir "pre_rounds"
            ;;
        "FedEM")
            n_learners=3
            set_inner_dir "sampling_rate"
            set_inner_dir "m"
            ;;
        "FedProx")
            set_inner_dir "mu"
            optimizer="prox_sgd"
            ;;
        "pFedMe")
            set_inner_dir "mu"
            ;;
        "FuzzyFL")
            set_inner_dir "pre_rounds"
            set_inner_dir "fuzzy_m"
            inner_dir+="_a_0.8"
            ;;
        "L2SGD")
            set_inner_dir "mu"
            set_inner_dir "comm_prob"
            ;;
        *)
            ;;
    esac

    if [[ ${parameters[minibatch]} ]]; then
        log_type="batch"
    fi 

    local log_dir="logs/$dataset/$log_type/$algo/${algo}_lr_${lr}${inner_dir}"
    local out_file="${dataset}_${algo}_lr_${lr}${inner_dir}.log"


    mkdir -p $log_dir  

    echo $log_dir
    echo $out_file
    # echo $extra_args_str 

    python run_experiment.py $dataset $algo --n_rounds 200 --n_learners $n_learners --bz 64 --lr $lr --log_freq 2 --optimizer $optimizer --local_steps 2 --lr_scheduler cosine_annealing --seed 1234 --verbose 1 ${extra_args_str} --logs_root $log_dir  > "$log_dir/$out_file"
}

run_gd() {
    local dataset="$1"
    shift
    run "$dataset" "$@"
}

run_batch() {
    local dataset="$1"
    shift
    run "$dataset" "$@" --minibatch 
}


 


