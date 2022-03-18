import fire
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm

plt.style.use("seaborn")
sns.set_theme(style="whitegrid")


def plot_many_returns(dir="data/", max_n=1000, pick_random=True, inflation_percent=0.0):
    files = natsorted([f for f in os.listdir(dir) if f.endswith(".csv")])
    # TODO randomly subsample instead of taking first max_n?
    if pick_random:
        files = random.sample(files, max_n)
    else:
        files = files[:max_n]

    n_simulations = len(files)

    print("Reading csv output files...")
    df_all = pd.DataFrame()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

    for i, f in enumerate(tqdm(natsorted(files))):
        df = pd.read_csv(os.path.join(dir, f), index_col=0, header=None).T[:-1]  # last is NaN, drop it
        idx = f.replace("output", "").replace(".csv", "")
        plt.plot(df['Values'], label=idx)

        dfi = pd.DataFrame(df['Values']).rename(columns={'Values': idx})
        df_all = pd.concat([df_all, dfi], axis=1)

    last_row = df_all.tail(1)
    mean_val = float(last_row.mean(axis=1))
    max_idx = last_row.idxmax(axis=1)
    max_val = float(last_row.max(axis=1))
    min_idx = last_row.idxmin(axis=1)
    min_val = float(last_row.min(axis=1))

    print(f"Mean capital value:    {mean_val}")
    print(f"Maximum capital value: {max_val} (simulation {max_idx})")
    print(f"Minimum capital value: {min_val} (simulation {min_idx})")

    print(df_all)
    store = pd.HDFStore(os.path.join(dir, 'store.h5'))
    store['df'] = df_all  # save it to HDF5 format

    ## Format the plot
    plt.title(f"Fund value over time (N={n_simulations})")
    plt.xlabel("Time (Months)")
    plt.ylabel("Fund value")
    if len(files) < 20: plt.legend()
    initial_capital = df_all.iloc[0][0]
    plt.plot([initial_capital] * len(df_all), label="Starting amount", color='r', linewidth=5)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "MC_capital_overview.png"))

    # find number of simulations where we end up lower than initial capital, corrected for inflation
    N_years = (len(df_all) - 1) / 12
    min_amount = initial_capital * (1 + inflation_percent / 100.0) ** N_years
    count_bad = np.sum(df_all.tail(1).values < min_amount)

    print(f"Out of {n_simulations} simulations, {count_bad} ended up with "
          f"less than the initial amount corrected for inflation of {inflation_percent}%")

    ############################################################
    ## now create subplot with distribution of average return ##
    ############################################################
    returns = df_all.iloc[-1] / df_all.iloc[0] - 1
    returns.sort_values(inplace=True)
    returns_percent = 100 * returns

    def annual_return(cumulative_return):
        ## cumulative return should be in percentage. Eg if we end with 121 and started with 100, it should be 0.21
        return (1 + cumulative_return) ** (1 / N_years) - 1

    # need to
    annual_returns = returns.apply(annual_return)
    annual_returns_percent = 100 * annual_returns

    extraticks_0 = returns_percent.quantile([.25, 0.5, .75]).values.tolist()
    extraticks_1 = annual_returns_percent.quantile([.25, 0.5, .75]).values.tolist()

    opts = dict(inner="quartile", bw=0.15)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    sns.violinplot(data=returns_percent, ax=axs[0], **opts)
    axs[0].set_title(f"Total returns across {N_years} years")
    axs[0].set_ylabel(f"Total return (%)")
    axs[0].set_yticks(axs[0].get_yticks().tolist() + extraticks_0)

    sns.violinplot(data=annual_returns_percent, ax=axs[1], **opts)
    axs[1].set_title(f"Annualized returns across {N_years} years")
    axs[1].set_ylabel(f"Annualized return (%)")
    axs[1].set_yticks(axs[1].get_yticks().tolist() + extraticks_1)

    # make quartile numbers bold
    for tick in axs[0].get_yticklabels()[-len(extraticks_0):]:
        tick.set_fontsize(12)
        tick.set_fontweight("heavy")

    for tick in axs[1].get_yticklabels()[-len(extraticks_1):]:
        tick.set_fontsize(12)
        tick.set_fontweight("heavy")
    plt.savefig(os.path.join(dir, "MC_returns_overview.png"))

    plt.show()


def plot_returns(csv_file="data/output.csv"):
    dirpth = os.path.dirname(csv_file)
    df = pd.read_csv(csv_file, index_col=0, header=None).T
    print(df)
    print("Final value: ", df.iloc[-2]['Values'])

    def plot_values():
        plt.plot(df['Values'])
        plt.title("Fund value over time")
        plt.xlabel("Time (Months)")
        plt.ylabel("Fund value")
        plt.tight_layout()
        plt.savefig(os.path.join(dirpth, "values.png"))

    def plot_returns():
        plt.figure()
        plt.plot(df['Returns'][1:])
        plt.title("Montly Returns over time")
        plt.xlabel("Time (Months)")
        plt.ylabel("Montly return")
        plt.tight_layout()
        plt.savefig(os.path.join(dirpth, "monthly_returns.png"))

    plot_values()
    plot_returns()

    plt.show()


if __name__ == "__main__":
    fire.Fire()
