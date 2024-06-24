import pandas as pd
from pathlib import Path
import json
import math

INPUT_DIR = "./OUTPUT/FTT/"

def import_data(file_path):
    df = pd.read_csv(file_path)
    return df

def import_metadata(file_path):
    metadata = json.load(open(file_path))
    return metadata

def calc_HRR(C, delta_P, T_c, O2_initial, O2_final, t):
    # calculate heat release rate
    hrr = 13100 * 1.1 * C * math.sqrt(delta_P/T_c) * ((O2_initial - O2_final*t)/(1.105 - 1.5*O2_final*t))
    return hrr

data = import_data(INPUT_DIR + "19010001_data.csv")
metadata = import_metadata(INPUT_DIR + "19010001_metadata.json")

def process(row):
    C = float(metadata["c-factor_si_units"])
    delta_P = float(metadata["pressure_rise_pa"])



# find HRR, other parameters
data.apply(process, axis=1)
