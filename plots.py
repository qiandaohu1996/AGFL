""""""
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
    print("tag",tag_)
    fig, ax = plt.subplots(figsize=(24, 20))

    for method in os.listdir(path_):
        print("\nmethod  ", method)
        if tag_ in ["Train/Loss", "Train/Metric"]:
            mode = "train"
        elif tag_ in ["Test/Loss", "Test/Metric"]:
            mode = "test"
        else:
            print(f"Invalid tag: {tag_}. Skipping...")
            continue
        method_path = os.path.join(path_, method, mode)

        print("method_path", method_path)
        
        # Check the correct tag depending on mode
        for task in os.listdir(method_path):
            if task == "global" :
                task_path = os.path.join(method_path, task)
                print("task_path  ",task_path)
                ea = EventAccumulator(task_path).Reload()
                # print("ea  " ,type(ea))
                try:
                    tag_values = []
                    steps = []
                    for event in ea.Scalars(tag):
                        tag_values.append(event.value)
                        steps.append(event.step)

                    if method in LEGEND:
                        ax.plot(
                            steps,
                            tag_values,
                            linewidth=5.0,
                            # marker=MARKERS[method],
                            markersize=20,
                            markeredgewidth=5,
                            label=f"{LEGEND[method]}",
                            color=COLORS[method]
                        )
                except KeyError:
                    print(f"Tag '{tag}' not found in {task_path}. Skipping...")
                    continue

    ax.grid(True, linewidth=2)

    ax.set_ylabel(AXE_LABELS[tag_], fontsize=50)
    ax.set_xlabel("Rounds", fontsize=50)

    ax.tick_params(axis='both', labelsize=25)
    ax.legend(fontsize=60)

    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f"{FILES_NAMES[tag_]}")
    plt.savefig(fig_path, bbox_inches='tight')



if __name__ == "__main__":
 
    path = "D:/AGFL-main/logs/emnist_pathologic_cl20/lr0.1"
    for tag in TAGS:
        make_plot(path_=path, tag_=tag, save_path='./figures/114/')