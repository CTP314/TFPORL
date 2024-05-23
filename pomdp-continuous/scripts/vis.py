import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid", {"grid.linestyle": "--"})
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 13  # 10
plt.rcParams["axes.grid"] = True
plt.rcParams["legend.loc"] = "best"
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.offset_threshold"] = 1
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True

import os
import sys
sys.path.append(os.getcwd())

from utils.vis_tool import walk_through
from itertools import product


def reduce_shared_encoder(shared_encoder, freeze_critic, freeze_all):
    if not shared_encoder:
        return "sep"
    if freeze_critic:
        return "sha-freeze"
    elif freeze_all:
        return "sha-freeze-all"
    else:
        return "sha-naive"


ni2022_results = {
    "ant-p": 348,
    "ant-v": 1113,
    "cheetah-p": 2693,
    "cheetah-v": 1980,
    "hopper-p": 2133,
    "hopper-v": 1495,
    "walker-p": 982,
    "walker-v": 121,
} # results in Ni et al., ICML 2022

def vis(log_path, output_dir, domains, palette):
    metric = "return"
    separate_legend = False

    hue = (
        "seq"
        # "encoder"
    )
    style = (
        None
    )

    def query_fn(flags):
        return True

    for env_type, env in product(["p", "v"], domains):
        base_path = f"{log_path}/pomdp/bullet"
        path = base_path + f"_{env_type}/{env}"
        end = 1.5e6
        df = walk_through(
            path,
            metric,
            query_fn,
            start=0,
            end=end,
            steps=100,
            window=10,
        )
        df = df.fillna(False)

        # custom functions to reduce flags
        df["encoder"] = df.apply(
            lambda row: reduce_shared_encoder(
                row["shared_encoder"], row["freeze_critic"], row["freeze_all"]
            ),
            axis=1,
        )
        df["seq"] = df["config_seq.model.seq_model_config.name"].str.upper()

        ans = sns.lineplot(
            data=df,
            x="env_steps",
            y=metric,
            palette=palette,
            hue=hue,
            hue_order=['LRU', 'GPT', 'LSTM'],
            style=style,
            style_order=np.sort(df[style].unique()) if style is not None else None,
        )
        if hue == "encoder":
            plt.axhline(y=ni2022_results[f"{env}-{env_type}"], label="sep-Ni2022")
        if "loss" in metric:
            ans.set_yscale("log")

        if separate_legend:
            ans.legend().set_visible(False)
        else:
            ans.legend(framealpha=0.2)  # must use the returned ans

        plt.xlim(0, end)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))  # default [-5, 6]
        plt.title(f"{env.title()}-{env_type.title()}")
        final_values = df.groupby(['seq', 'submit_time', 'seed']).apply(lambda x: x.iloc[-1:].assign(**{metric: x[metric][-15:].mean()})).reset_index(drop=True)
        print(env, env_type)
        print(final_values.groupby('seq').agg({metric: ['mean', 'std']}))
        # print(final_values)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, path.replace("logs/", "").replace("/", "-"))
            + f"_{metric}_{hue}_{style}"
            + ("" if separate_legend else "_leg")
            + ".jpg",
            bbox_inches="tight",
            pad_inches=0.03,
        )  # default 0.1
        plt.show()
        plt.close()
              
def merge_log_dirs(log_dir_list):
    if os.path.exists("./logs_for_vis"):
        # os.removedirs("./logs_for_vis")
        os.system("rm -r logs_for_vis")
    os.makedirs("logs_for_vis", exist_ok=True)
    for log_dir in log_dir_list:
        os.system(f"cp -r {log_dir}/* logs_for_vis")
        
if __name__ == "__main__":
    merge_log_dirs(["logs_ant", "pybullet_jax_log"])
    palette = sns.color_palette("bright", 4)
    palette = [palette[3], palette[1], palette[2]]
    vis("logs_for_vis", "plts", ["ant"], palette)