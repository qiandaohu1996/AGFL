""""""
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
# sys.path
TAGS = [
    "Train/Loss",
    "Train/Metric",
    "Test/Loss",
    "Test/Metric",
    "Local_Test/Metric",
    "Local_Test/Loss"
]
TAGS = [
    "Train/Loss",
    "Train/Metric",
    "Test/Loss",
    "Test/Metric"
]
FILES_NAMES = {
    "Train/Loss": "train-loss.png",
    "Train/Metric": "train-acc.png",
    "Test/Loss": "test-loss.png",
    "Test/Metric": "test-acc.png",
    "Local_Test/Metric": "local-test-acc.png",
    "Local_Test/Loss": "local-test-loss.png"}

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

 
def make_plot(path_, tag_, save_path):

    """
    :param path_: path of the logs directory, `path_` should contain sub-directories corresponding to methods
        each sub-directory must contain a single tf events file.
    :param tag_: the tag to be plotted, possible are "Train/Loss", "Train/Metric", "Test/Loss", "Test/Metric"
    :param save_path: path to save the resulting plot
    """
    # print("path_",path_)
    # print("\ntag",tag_)
    fig, ax = plt.subplots(figsize=(24, 20))
    mode = tag_.split("/")[0].lower()
    dataset=path_.split("\\")[-2]
    # print("mode ",mode)
    print("\ndataset   ",dataset)

    for method in os.listdir(path_):
        # print("\nmethod  ", method)
        if method == "pFedMe":
            continue
        params_dir = os.path.join(path_, method)
        # print("\nparams_dir  ", params_dir)

        for param_dir in os.listdir(params_dir):

            # if  param_dir == "FedProx_lr_0.1_mu_0.5":
            #     continue
            # if  param_dir == "FedProx_lr_0.1_mu_1":
            #     continue
            if os.path.isfile(os.path.join(params_dir, param_dir)) :
                continue
            # if "lr_0.03" in param_dir:
            #     continue
            method_path = os.path.join(params_dir, param_dir, mode)

            print("method_path", method_path)
            print("param_dir ", param_dir)
            
            # Check the correct tag depending on mode
            for task in os.listdir(method_path):
                if task == "global" :
                    task_path = os.path.join(method_path, task)
                    # print("task_path  ",task_path)
                    ea = EventAccumulator(task_path).Reload()
                    # print("ea  " ,type(ea))
                    try:
                        tag_values = []
                        steps = []
                        for event in ea.Scalars(tag):
                            tag_values.append(event.value)
                            steps.append(event.step)
                        # window_size = 1
                        # smoothed_tag_values = np.convolve(tag_values, np.ones(window_size)/window_size, mode='valid')
                        param_dir = param_dir.replace("clustered", "CFL")
                        # print("param_dir 2 ", param_dir)

                        minilabel= (param_dir.split("_"))
                        # print("minilabel", minilabel)
                        label=" ".join(minilabel)
                        # print("label", label)

                        # if method in LEGEND:
                        ax.plot(
                            steps[1:],
                            tag_values[1:],
                            linewidth=5.0,
                            # marker=MARKERS[method],
                            markersize=20,
                            markeredgewidth=5,
                            label=label,
                            # color=COLORS[method],
                        )
                    except KeyError:
                        print(f"Tag '{tag}' not found in {task_path}. Skipping...")
                        continue

            ax.grid(True, linewidth=2)

            ax.set_ylabel(AXE_LABELS[tag_], fontsize=40)
            ax.set_xlabel("Rounds", fontsize=40)

            ax.tick_params(axis='both', labelsize=25)
            ax.legend(fontsize=30)

            os.makedirs(save_path, exist_ok=True)
            fig_path = os.path.join(save_path, f"{dataset}_{FILES_NAMES[tag_]}")
            plt.savefig(fig_path, bbox_inches='tight')

if __name__ == "__main__": 
 
    current_dir = os.getcwd()

    # datasets=["emnist_component4", "emnist", "emnist_pathologic_cl20","femnist","cifar100_s0.25","cifar100"]
    datasets=["cifar100_s0.25"]
    for dataset in datasets:
        relative_path = os.path.join("logs", dataset,"batch")

        path = os.path.join(current_dir, relative_path)
        
        for tag in TAGS:
            make_plot(path_=path, tag_=tag,  save_path=f'./figures/{dataset}/')

