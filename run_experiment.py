"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
import sys
from utils.utils import *
from utils.constants import *
from utils.args import *
import time

from torch.utils.tensorboard import SummaryWriter


def init_clients1(args_, root_path, logs_root):
    # ...
    print("===> Building data iterators..")

    train_iterators, val_iterators, test_iterators =\
        get_loaders(
            type_=get_loader_type(args_.experiment),
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation
        )
    print("===> Initializing clients..")
    clients_ = []   
    for task_id, (train_iterator, test_iterator) in \
            enumerate(zip(train_iterators, test_iterators), start=1):
        if train_iterator is None or test_iterator is None:
            continue
            
        client = init_single_client(args_, train_iterator, test_iterator, task_id, logs_root)
        clients_.append(client)
    
    return clients_
@memory_profiler
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def init_clients(args_, root_path, logs_root):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")

    train_iterators, val_iterators, test_iterators =\
        get_loaders(
            type_=get_loader_type(args_.experiment),
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):
        # print("task_id ",task_id)
        if train_iterator is None or test_iterator is None:
            continue
            
        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                name=args_.experiment,
                method=args_.method,
                adaptive_alpha=args_.adaptive_alpha,
                alpha=args_.alpha,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
            )

        logs_path = os.path.join(logs_root, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            fuzzy_m=args_.fuzzy_m,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)
    return clients_

def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    with segment_timing("init clients "):

        clients = init_clients(
            args_,
            root_path=os.path.join(data_dir, "train"),
            logs_root=os.path.join(logs_root, "train")
        )

    print("==> Test Clients initialization..")
    with segment_timing("init test clients"):

        test_clients = init_clients(
            args_,
            root_path=os.path.join(data_dir, "test"),
            logs_root=os.path.join(logs_root, "test")
        )


    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)


    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test","client_avg")
    os.makedirs(logs_path, exist_ok=True)
    local_test_logger = SummaryWriter(logs_path)
    

    # with segment_timing("init global_learners_ensemble"):

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            method=args_.method,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            alpha=args_.alpha,
            adaptive_alpha=args_.adaptive_alpha,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu

        )

   
    # with segment_timing("init aggregator"):
    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]
        aggregator =\
            get_aggregator(
                aggregator_type=aggregator_type,
                clients=clients,
                global_learners_ensemble=global_learners_ensemble,
                lr_lambda=args_.lr_lambda,
                lr=args_.lr,
                q=args_.q,
                mu=args_.mu,
                fuzzy_m=args_.fuzzy_m,
                communication_probability=args_.comm_prob,
                sampling_rate=args_.sampling_rate,
                pre_rounds=args_.pre_rounds,
                log_freq=args_.log_freq,
                global_train_logger=global_train_logger,
                global_test_logger=global_test_logger,
                local_test_logger=local_test_logger,
                test_clients=test_clients,
                single_batch_flag=args_.minibatch,
                verbose=args_.verbose,
                seed=args_.seed
            )
    torch.cuda.empty_cache()
    # print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0

    print("\n========Training begins======== \n", time.ctime())
    with segment_timing("Training total time "):
        while current_round <= args_.n_rounds:
            if current_round<5 or current_round ==50:
                Profile_Memory=True
            else: 
                Profile_Memory=False
            print(f"===========train at {current_round} round===========")
            with segment_timing(f"{current_round} round executing time "):
                aggregator.mix()
 
            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round


    if "save_path" in args_:
        save_root = os.path.join(args_.save_path)

        os.makedirs(save_root, exist_ok=True)
        aggregator.save_state(save_root)



if __name__ == "__main__":
    PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 获取当前命令行命令
    command = ' '.join(sys.argv)

    # 打印当前命令行命令
    print("command line: ", command)

    start_time=time.time()
    print("start at ", time.ctime())

    args = parse_args()
    run_experiment(args)

    end_time = time.time()

    print("end at ", time.ctime())
    
    print("command line: ", command)
    
    exec_time = end_time - start_time
    exec_time = time.strftime('%H:%M:%S', time.gmtime(exec_time))
    # 打印格式化后的时间
    print(f"\nexecution time {exec_time}")