import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import gaussian_filter1d
from utils.args import parse_args
import re



# DATA=["emnist", "emnist_c4", "femnist", "emnist_pathologic_cl20",   "cifar100_s0.25" "cifar10_n50",   "cifar100_n20"]
DATA=["emnist", "emnist_c4", "femnist", "emnist_pathologic_cl20"]

TAGS = [
    "Train/Loss",
    "Train/Metric",
    "Test/Loss",
    "Test/Metric",
    "Local_Test/Metric",
    "Local_Test/Loss"
]
 
FILES_NAMES = {
    "Train/Loss": "train-loss",
    "Train/Metric": "train-acc",
    "Test/Loss": "test-loss",
    "Test/Metric": "test-acc",
    "Local_Test/Metric": "local-test-acc",
    "Local_Test/Loss": "local-test-loss"}

AXE_LABELS = {
    "Train/Loss": "Train loss",
    "Train/Metric": "Train acc",
    "Test/Loss": "Test loss",
    "Test/Metric": "Test acc",
    "Local_Test/Metric": "Local_Test acc",
    "Local_Test/Loss" : "Local_Test loss" 
}
# LEGEND = {
#     "local": "Local",
#     "clustered": "Clustered FL",
#     "FedAvg_lr_0.1": "FedAvg",
#     "FedEM": "FedEM (Ours)",
#     "FedAvg_adapt": "FedAvg+",
#     "personalized": "pFedMe",
#     "FedProx": "FedProx",
#     "FuzzyFL_lr_0.03": "FuzzyFL",
# }
LEGEND = {
    "FedAvg": "FedAvg",
    # "FedEM": "FedEM",
    "FuzzyFL": "FuzzyFL",
    "FedProx": "FedProx",
    "pFedMe": "pFedMe",
    "clustered": "Clustered FL",
    "APFL": "APFL",
    "L2SGD": "L2SGD"
}

MARKERS = {
    "local": "x",
    "clustered": "s",
    "FedAvg": "h",
    "FedEM": "d",
    "APFL": "q",
    "AGFL": "p",
    "FuzzyFL": "s",
    "FedAvg_adapt": "4",
    "pFedMe": "X",
    "DEM": "|",
    "FedProx": "p"
}

COLORS = {
    "local": "tab:blue",
    "clustered": "tab:orange",
    "FedAvg": "tab:green",
    "FedEM": "tab:red",
    "FedAvg_adapt": "tab:purple",
    "pFedMe": "tab:brown",
    "DEM": "tab:pink",
    "APFL": "tab:pink",
    "FedProx": "tab:cyan",
    "FuzzyFL": "tab:red"
}

# FILTER_WORDS=[]
# CONTAIN_WORDS=[]
def check_list(lst):
    # 判断列表是否非空
    if not lst:
        return False
    
    # 判断列表中的字符串是否非空
    for s in lst:
        if not s  :
            return False
    return True

def to_dict(input_string):
    parts = input_string.split('_')[1:]   
    keys = parts[::2]
    values = parts[1::2]
    return dict(zip(keys, [float(v) for v in values]))

def replace_sample(input_string):
    return re.sub(r'samp(\d+\.\d+)', r'sampling \1', input_string)

def check_fuzzy_folder(folder_name):
    # filter_words=FILTER_WORDS
    # print("folder_name ",folder_name)
    # print("contain_words ",contain_words)
    # print("filter_words ",filter_words)

    parts = folder_name.split('_')[1:]

    parameters = ['_'.join(parts[i:i+2]) for i in range(0, len(parts), 2)]
    if check_list(FUZZY_CONTAIN_WORDS) == False:
        return check_folder(folder_name)
    if all(word in parameters for word in CONTAIN_WORDS) and \
        all(word in parameters for word in FUZZY_CONTAIN_WORDS) and \
        not any(word in parameters for word in FILTER_WORDS) :
        return True
    return False

def check_folder(folder_name):
    # filter_words=FILTER_WORDS
    # print("folder_name ",folder_name)
    # print("contain_words ",contain_words)
    # print("filter_words ",filter_words)

    if not check_list(CONTAIN_WORDS):
        if not any(word in parameters for word in FILTER_WORDS) :
            return True
    else:
        parts = folder_name.split('_')[1:]
        if "prob" in parts:
            parts.remove("prob")

        parameters = ['_'.join(parts[i:i+2]) for i in range(0, len(parts), 2)]
        # print(parameters)
        if all(word in parameters for word in CONTAIN_WORDS) and \
            not any(word in parameters for word in FILTER_WORDS) :
            return True
    
    return False

def check_method(method_name):
    if method_name in REPORT_METHODS and \
        method_name not in INVISBLE_METHODS:
        return True
    return False
    

def make_plot(path_, tag_, save_path):

    def _plot(method_path, param_dir):
        for task in os.listdir(method_path):
            if task == task_dir :
                task_path = os.path.join(method_path, task)
                ea = EventAccumulator(task_path).Reload()
                try:
                    tag_values = []
                    steps = []
                    for event in ea.Scalars(tag_):
                        tag_values.append(event.value)
                        steps.append(event.step)
                   
                    param_dir = param_dir.replace("clustered", "clustered FL")
                    param_dir = param_dir.replace("_lr_0.1", "")
                    param_dir = param_dir.replace("_pre_25", "")

                    minilabel= (param_dir.split("_"))[:-2]
                    # minilabel= (param_dir.split("_"))
                    if SMOOTH:
                        tag_values = gaussian_filter1d(tag_values, sigma=0.8)
                        
                    # label=" ".join(minilabel[:-2])
                    label=" ".join(minilabel)
                    
                    print("label", label)

                    ax.plot(
                        steps,
                        tag_values,
                        linewidth=4.5,
                        # markersize=20,
                        # markeredgewidth=5,
                        label=label,
                    )
                except KeyError:
                    print(f"Tag '{tag_}' not found in {task_path}. Skipping...")
                    continue
                except FileNotFoundError:
                    print(f"File{task_path} not found.")

    fig, ax = plt.subplots(figsize=(30, 25))
    mode = tag_.split("/")[0].lower()
    task_dir="global"
    title_samp=""
    if mode == "local_test":
        mode="test"
        task_dir="client_avg"
    dataset=path_.split("\\")[-2]
    
    for method in os.listdir(path_):
        print("\nmethod  ", method)

        if not check_method(method):
            print (check_method(method))
            continue
        method_dir = os.path.join(path_, method)
        print("\nmethod_dir ", method_dir)

        if method!="FuzzyFL":

            for param_dir in os.listdir(method_dir):
                print("\nparam_dir ", param_dir)

                if not check_folder(param_dir):
                    print("pass...")
                    continue
                if os.path.isfile(os.path.join(method_dir, param_dir)) :
                    continue
                method_path = os.path.join(method_dir, param_dir, mode)
                print("method path ", method_path)
                _plot(method_path, param_dir)

        else:
            for sampling_dir in os.listdir(method_dir):

                title_samp= replace_sample(sampling_dir)
                param_dirs= os.path.join(method_dir, sampling_dir)
                for param_dir in os.listdir(param_dirs):

                    print("\nparam_dir ", param_dir)
                    if os.path.isfile(os.path.join(param_dirs, param_dir)) :
                        continue
                    if not check_fuzzy_folder(param_dir) :
                        print("pass...")
                        continue
                
                    method_path = os.path.join(param_dirs, param_dir, mode)
                    print("method_path ", method_path)
                    _plot(method_path, param_dir)
            
        ax.grid(True, linewidth=2)
        ax.set_ylabel(AXE_LABELS[tag_], fontsize=40)
        ax.set_xlabel("Rounds", fontsize=40)

        ax.tick_params(axis='both', labelsize=25)
        ax.legend(fontsize=30)

        os.makedirs(save_path, exist_ok=True)
        extension=".png"
        fig_path = os.path.join(save_path, f"{dataset}_{FILES_NAMES[tag_]}{extension}")

        print("fig_path ", fig_path)
        title= dataset+" "+title_samp
        plt.title(title,fontsize=40)
        try:
            plt.savefig(fig_path, bbox_inches='tight')
            print("Figure saved successfully.")
        except Exception as e:
            print("Failed to save figure:", str(e))
 
def plot(datasets="emnist"): 
 
    current_dir = os.getcwd()
    '''
    # datasets=[ "emnist", "emnist_pathologic_cl20","femnist"]
    # datasets=["emnist_component4", "emnist", "emnist_pathologic_cl20","femnist","cifar100_s0.25","cifar100"]
    
    # datasets=["cifar100_s0.25"]
    # datasets=["emnist"]
    '''

    datasets=datasets.split(" ")
    for dataset in datasets:
        print(dataset)

        relative_path = os.path.join("logs", dataset,"batch")

        path = os.path.join(current_dir, relative_path)
        inder_dir=""
        inder_dir=CONTAIN_WORDS[0].replace("_","")

        inder_dir2="/"+"".join(FUZZY_CONTAIN_WORDS).replace("_","")
        # if len(CONTAIN_WORDS)==1:
        #     inder_dir+="/Fuzzy"

        save_path=f'./figures/{dataset}/{inder_dir}{inder_dir2}'
        for tag in TAGS:
            print("\ntag ", tag)
            make_plot(path_=path, tag_=tag,  save_path=save_path)
    
    print(f'\nIf Plotting completed, saved in {save_path}')
        

if __name__ == "__main__": 

    # datasets="femnist"
    # datasets="emnist50"
    # datasets="emnist50_c4"
    # datasets="emnist"
    datasets="cifar100"
    # datasets="synthetic00"
    # REPORT_METHODS=["pFedMe","FedAvg","L2SGD","FedEM","FedProx","FuzzyFL"]
    REPORT_METHODS=["FedAvg","FedEM","FuzzyFL"]
    INVISBLE_METHODS=[""]
    SMOOTH=True
    # FILTER_WORDS=["samp_0.1", "pre_1", "samp_0.2","samp_1",,"mt_0.5" "pre_50"]
    FILTER_WORDS=[]
    CONTAIN_WORDS=["samp_0.2"]
    # FUZZY_CONTAIN_WORDS=["pre_1","m_1.75","trans_0.75","sch_cosine"]
    FUZZY_CONTAIN_WORDS=[]
    # FUZZY_CONTAIN_WORDS=["pre_25","m_1.75","trans_0.75","mt_0.8"]
    # print(globals())
    # print(locals())
    plot(datasets)



