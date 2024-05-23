import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.vis_tool import walk_through
import warnings

warnings.filterwarnings("ignore")

separate_legend = True  # False  #

ablation = ""  # "gpt" # ,"lstm" #
hue = (
    # "config_rl.algo"
    "seq"
    # "n_layer"
)
style = None

def query_fn(flags):
    return True

def get_data(log_path, domain, tasks):
    task_fn = lambda task: task.title()
    infos_fn = lambda task: zip([25], [1e6])
    path_fn = lambda task, env_len: f"{log_path}/{domain}_{task}/{env_len}"
    env_type_fn = lambda task, l: l

    dfs = []

    for task in tasks:
        infos = infos_fn(task)
        for env_len, end in infos:
            env_name = task_fn(task)
            path = path_fn(task, env_len)
            metric = "return"
            
            df = walk_through(
                path,
                metric,
                query_fn,
                start=0,
                end=end,
                steps=300,
                window=10,
                is_regular=domain == 'regular',
            )
            df = df.fillna(False)

            # custom functions to reduce flags
            df["seq"] = df["config_seq.model.seq_model_config.name"].str.upper()
            print(df["seq"].unique())
            if "config_seq.model.seq_model_config.n_layer" in df:
                df["n_layer"] = df["config_seq.model.seq_model_config.n_layer"]
            df = df.assign(
                **{'env': task, 'env_type': env_type_fn(task, str(env_len))}
            )
            print(task, env_len, env_type_fn(task, str(env_len)))
            if task == 'Passive T-Maze':
                df['success'] = df['return'].apply(lambda x: max(x, 0))
            dfs.append(df)
            
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df['env_name'] = df['env'].map(lambda x: x.replace('_', ' ').title() if x != 's5' else '$\mathsf{NC}^1$ Complete')
    df['env_type'] = df['env_type'].map(lambda x: x.title())
    return df

def plot(df, domain):
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (12, 3)
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 13  # 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams["axes.formatter.offset_threshold"] = 1
    # plt.rcParams["font.size"] = 8
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Liberation Serif"]
    plt.rcParams["text.usetex"] = True
    
    df_plot = df.copy()
    
    n_head = '$H$'
    n_layer = '$L$'
    n_hidden = '$D$'
    f, axes = plt.subplots(1, 3)
    
    
    df_plot[n_head] = df_plot['config_seq.model.seq_model_config.n_head']
    df_plot[n_layer] = df_plot['n_layer']
    df_plot[n_hidden] = df_plot['config_seq.model.seq_model_config.hidden_size']
    
    for ax, ablation in zip(axes, [n_head, n_layer, n_hidden]):
        sns.lineplot(
            data=df_plot,
            x="env_steps",
            y="return",
            hue=ablation,
            # style=style,
            ax=ax,
            # ci=95,
            # markers=True,
            # dashes=False,
            # err_style="band",
            palette="colorblind",
        )
        # ax.set_xlabel(ablation)
        # ax.set_ylabel("Return")
        # ax.set_title(f"{ablation} Ablation")
        # ax.legend().set_visible(False)
    
    plt.savefig(f"ablation_{domain}.jpg", bbox_inches='tight')

if __name__ == "__main__":
    # f('logs_old', 'logs')
    domain = "regular"
    tasks = ["s5"]
    # palette = sns.color_palette("bright", 4)
    # palette = [palette[3], palette[1], palette[2]]
    df = get_data("ablation", domain, tasks)
    print(df.columns)
    plot(df, domain)
    
