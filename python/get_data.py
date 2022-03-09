import yfinance as yf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import fire
import seaborn as sns
import os

sns.set_style("ticks")

def get_data_SP500():
    print("- Downloading S&P500 data from Yahoo finance...")
    data = yf.download("^GSPC",'1928-01-01','2022-01-01')
    df = data[['Adj Close']]
    df_monthly = df.resample('1M').mean()

    def plot_data(df, df_monthly):
        print("- Plotting...")
        fig, ax = plt.subplots(figsize=(16,10))
        df.plot(logy=True, ax=ax, y='Adj Close', label='daily')

        df_monthly.plot(logy=True, ax=ax, y='Adj Close', label='monthly', x_compat=True)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

        plt.grid(which='both', axis='both', color='grey', linestyle='-', linewidth=1, alpha=0.5)
        fig.subplots_adjust(bottom=0.1, wspace=0.33)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),fancybox=False, shadow=False, ncol=2)
        plt.title("Daily and Monthly returns of S&P500")
        plt.tight_layout()
        os.makedirs("data")
        plt.savefig("data/SP500_monthly_returns.png")
        print("Saved plot of historical returns to 'data/' directory")
        #plt.show()

    plot_data(df, df_monthly)

    ## now extract monthly returns
    print("- Generating CSV file with monthly returns...")
    monthly_returns = 100 * df_monthly.pct_change()

    # simplify and clean up
    monthly_returns.rename(columns={"Adj Close": "returns"}, inplace=True)
    monthly_returns.index = monthly_returns.index.to_period("M")

    # save to disk
    monthly_returns.to_csv("data/SP500_monthly_returns.csv")
    print("- Saved CSV of monthly historical returns to 'data/' directory")

    print("Done!")



if __name__ == "__main__":
    fire.Fire()