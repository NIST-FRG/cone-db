import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from FTT import parse_data


def compare():
    # get all CSV files in input folder that end with _red.csv
    # (reduced data files)
    paths = list(Path("./DATA/FTT").glob("*_red.csv"))

    count = len(paths)
    print(count)

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
        axs[idx].set_title(f"{path.stem}")
        idx += 1

    plt.show()


def plot_all():
    # get all CSV files in input folder that end with _red.csv
    # (reduced data files)
    paths = list(Path("./OUTPUT/FTT").glob("*.csv"))

    # count = len(paths)

    # x_idx = 0
    # y_idx = 0

    # height = 5
    # width = round(count / height)

    # fig, axs = plt.subplots(height, width, constrained_layout=True)


    # for each file, read it in as a pandas dataframe
    for path in paths:
        # read in the corresponding output file from script
        processed_df = pd.read_csv(path)
        print(f"Plotting: {path.stem}")

        # if y_idx >= height:
        #     y_idx = 0
        #     x_idx += 1

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        ax.plot(processed_df["HRR (kW/m2)"])
        ax.plot(processed_df["HRR (kW/m2)"].rolling(10, center=True).mean())
        ax.set_title(f"{path.stem}")
        plt.savefig(f"./GRAPHS/{path.stem.replace("_data", "")}.png")
        plt.close()

        # y_idx += 1

    


# plot_all()
compare()
