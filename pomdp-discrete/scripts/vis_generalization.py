import os
import pandas as pd

def get_data(log_path):
    df_list = []
    for root, dirs, files in os.walk(log_path):
        if 'progress_eval.csv' in files:
            _, task, len, seq_model, exp  = root.split('/')
            if len == '25':
                print(f"{root}/progress_eval.csv")
                df = pd.read_csv(f"{root}/progress_eval.csv")
                df['task'] = task
                df['seq_model'] = seq_model
                df['exp'] = exp
                df_list.append(df)
            
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df

import matplotlib.pyplot as plt
import seaborn as sns


def vis(domain, data: pd.DataFrame, palette):
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (5, 3)
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

    plot_data = data.copy()
    plot_data['env_name'] = plot_data['task'].map(lambda x: x.replace('_', ' ').replace('regular ', '').title() if x != 's5' else '$\mathsf{NC}^1$ Complete')
    plot_data['seq'] = plot_data['seq_model'].str.upper()
    metric = 'success'
    env_order = ['Even Pairs', 'Parity', '$\mathsf{NC}^1$ Complete']
    plt.subplots_adjust(hspace=100, wspace=0.4)
    g = sns.FacetGrid(
        plot_data, 
        col="env_name",  
        margin_titles=True,
        # row_order=env_order,
        # sharex=False,
        # sharey=False,
    )
    times = plot_data.length.unique()
    print(times)
    ans = g.map(sns.barplot, "length", metric, "seq", hue_order=['LRU', 'GPT', 'LSTM'], errorbar=None, palette=palette)
    ans.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    # plt.xlim(0, None)
    # plt.ylim(0, None)
    # plt.tight_layout()
    # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))  # default [-5, 6]
    sns.move_legend(
        ans, "lower center",
        bbox_to_anchor=(0.5, 1), ncol=3, title=None, frameon=True,
    )
    plt.savefig(
        f"{domain}_plts/{domain}-{metric}-length"
        + ".jpg",
        bbox_inches="tight",
        pad_inches=0.03,
    )  # default 0.1
    plt.close()
    # 
    
    # sns.catplot(x='length', y='success', hue='seq_model', data=data, kind='bar', row='task', errorbar=None)

    # sns.barplot(x='length', y='success', hue='seq_model', data=data, row)

    # plt.savefig('output.jpg', dpi=300)  # 指定dpi可以设置输出图像的分辨率
    # plt.show()    

if __name__ == '__main__':
    data = get_data('logs_old')
    palette = sns.color_palette("bright", 4)
    palette = [palette[3], palette[1], palette[2]]
    domain = "regular"
    vis(domain, data, palette)