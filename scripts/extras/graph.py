import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
import scipy.signal as signal
import json
import numpy as np

# might be somewhat outdated - needs to be updated

INPUT = Path("./parse/output/FTT")
OUTPUT = Path("./extras/graphs/FTT")

def compare_FTT():
    # get all CSV files in input folder that end with _red.csv
    # (reduced data files)
    paths = list(INPUT.glob("*_red.csv"))

    count = len(paths)

    idx = 0

    fig, axs = plt.subplots(count, constrained_layout=True)
    fig.set_size_inches(8, 3 * count)
    # fig.tight_layout()

    # for each file, read it in as a pandas dataframe
    for path in paths:
        orig_df = pd.read_csv(path, encoding="cp1252", skiprows=1)

        # read in the corresponding output file from script
        processed_df = pd.read_csv(
            str(path).replace("_red", "_data").replace("DATA", "OUTPUT")
        )

        # print it out
        # print(orig_df["HRR (kW/m²)"])
        # print(processed_df["HRR (kW/m2)"])
        axs[idx].plot(orig_df["HRR (kW/m²)"], label="FTT")
        axs[idx].plot(processed_df["HRR (kW/m2)"], label="Script")
        axs[idx].legend(loc="upper left")
        axs[idx].set_title(f"Test: {path.stem.replace("_red", "")}")
        axs[idx].xaxis.set_major_locator(ticker.MultipleLocator(50))

        
        idx += 1

    plt.show()


def plot_all(folder_name):
    # get all CSV files in input folder that end with _red.csv
    # (reduced data files)
    paths = list(INPUT.rglob("*.csv"))

    # count = len(paths)

    # x_idx = 0
    # y_idx = 0

    # height = 5
    # width = round(count / height)

    # fig, axs = plt.subplots(height, width, constrained_layout=True)

    def plot(df, metadata):
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 5)
        # fig.tight_layout()
        mlr = processed_df["MLR (g/s)"]

        ax2 = ax.twinx()
        o2 = processed_df["O2 (%)"]
        o2 = signal.savgol_filter(o2, 31, 3)
        do2 = np.gradient(o2)
        ax2.plot(do2, color="r", label="Derivative of O2 (%)")
        ax2.axhline(y=0, color="black", linewidth = 1)
        

        ax.plot(o2, label="O2 (%)")
        ax.set_xlim(0, 200)
        
        ax.set_ylim(19, 21)


        ax.axhline(y=0, color='b', linestyle='--', linewidth=3)
        ax.axvline(x=metadata["time_to_ignition_s"], color='y', linestyle='--', linewidth=3)

        # ax.plot(mlr, label="Original")

        # ax.plot(signal.savgol_filter(mlr, 31, 3), label="Savitzky-Golay (9 frames, order 5)")
        
        ax.set_title(f"{path.stem}")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        plt.savefig(Path(OUTPUT / f"{path.stem.replace("_data", "")}.png"))
        plt.close()

    # for each file, read it in as a pandas dataframe
    for path in paths:
        # read in the corresponding output file from script
        processed_df = pd.read_csv(path)
        print(f"Plotting: {path.stem}")

        metadata = json.load(open(Path(OUTPUT / f"{folder_name}/{path.stem.replace("_data", "_metadata")}.json")))

        try:
            plot(processed_df, metadata)
        except:
            print(f"Error plotting {path.stem}")
            continue

    

if __name__ == "__main__":
    plot_all("FTT")
    # compare_FTT()
