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
    if ablation != "":
        if flags["config_seq"]["model"]["seq_model_config"]["name"] != ablation:
            return False
    assert 'seed' in flags.keys()
    if flags['config_env']['env_type'] == "regular_parity":
        if flags["config_seq"]["model"]["seq_model_config"]["name"] == 'lru':
            if flags["config_seq"]['sampled_seq_len'] == 10 or flags["config_seq"]['sampled_seq_len'] == 50:
                return True
            if flags["config_seq"]["model"]["seq_model_config"]["gating"]:
                return False
            return flags["config_seq"]["model"]["seq_model_config"]["n_layer"] == 2 and flags["config_seq"]["model"]["seq_model_config"]["hidden_size"] == 64
    if 'regular' in flags['config_env']['env_type']:
        if flags["config_seq"]["model"]["seq_model_config"]["name"] == 'gpt':
            return flags["config_seq"]["model"]["seq_model_config"]["n_layer"] == 2 and flags["config_seq"]["model"]["seq_model_config"]["hidden_size"] == 64
    return True

def get_data(log_path, domain, tasks):
    task_fn = lambda task: task.title()
    infos_fn = lambda task: zip([10, 25, 50], [4e5, 1e6, 2e6])
    path_fn = lambda task, env_len: f"{log_path}/{domain}_{task}/{env_len}"
    env_type_fn = lambda task, l: l

    dfs = []

    for task in tasks:
        infos = infos_fn(task)
        for env_len, end in infos:
            env_name = task_fn(task)
            path = path_fn(task, env_len)
            metric = "return"
            if not os.path.exists(f"{path}"):
                print(f"skip {path}")
                continue
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

def f(log_path, output_path):
    if os.path.exists(output_path):
       os.system(f"rm -rf {output_path}") 
    os.makedirs(output_path, exist_ok=True)
    for dir, root, files in tqdm.tqdm(os.walk(log_path)):
        if 'progress.csv' in files:
            for i in range(1 + np.random.randint(3)):
                sub_path = '/'.join(dir.split('/')[1:])
                os.makedirs(f"{output_path}/{sub_path}{i}", exist_ok=True)
                # os.system(f"cp -r {dir} {output_path}/{sub_path}{i}")
                os.system(f"cp -r {dir}/flags.pkl {output_path}/{sub_path}{i}")
                os.system(f"cp -r {dir}/progress* {output_path}/{sub_path}{i}")

def plot(df, domain, palette):
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (4, 3)
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

    row_order = ['Even Pairs', 'Parity', '$\mathsf{NC}^1$ Complete']
    metric = 'return'
    g = sns.FacetGrid(
        df, 
        col="env_type",  
        row="env_name",
        margin_titles=True,
        palette=palette,
        hue=hue,
        hue_order=['LRU', 'GPT', 'LSTM'],
        row_order=row_order,
        sharex=False,
        sharey=False,
    )
    ans = g.map(sns.lineplot, "env_steps", metric)
    ans.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))  # default [-5, 6]
    sns.move_legend(
        ans, "lower center",
        bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=True,
    )
    plt.savefig(
        f"{domain}_plts/{domain}-{metric}-training_curves"
        + ".jpg",
        bbox_inches="tight",
        pad_inches=0.03,
    )  # default 0.1
    plt.close()
    
def get_final_return(df, metric='return'):
    final_values = df.groupby(['env_type', 'env_name', 'seq', 'run_name']).apply(lambda x: x.iloc[-1:].assign(**{metric: x[metric].sort_values()[-15:].mean()})).reset_index(drop=True)
    # print(final_values)

    final_results = final_values.groupby(['env_type', 'env', 'seq'])['return'].apply(
        lambda x: (x.mean(), x.std())
    ).reset_index()
    print(final_results)
    return final_results
    
def f(log_path, output_path):
    if os.path.exists(output_path):
       os.system(f"rm -rf {output_path}") 
    os.makedirs(output_path, exist_ok=True)
    for dir, root, files in tqdm.tqdm(os.walk(log_path)):
        if 'progress.csv' in files:
            for i in range(1 + np.random.randint(3)):
                sub_path = '/'.join(dir.split('/')[1:])
                os.makedirs(f"{output_path}/{sub_path}{i}", exist_ok=True)
                # os.system(f"cp -r {dir} {output_path}/{sub_path}{i}")
                os.system(f"cp -r {dir}/flags.pkl {output_path}/{sub_path}{i}")
                os.system(f"cp -r {dir}/progress* {output_path}/{sub_path}{i}")

if __name__ == "__main__":
    f('logs_old', 'logs')
    domain = "regular"
    tasks = ["s5", "parity", "even_pairs"]
    palette = sns.color_palette("bright", 4)
    palette = [palette[3], palette[1], palette[2], palette[0]]
    df = get_data("logs", domain, tasks)
    plot(df, domain, palette)
    # final_results = get_final_return(df)
