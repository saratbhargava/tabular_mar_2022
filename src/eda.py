import argparse

import numpy as np
import pandas as pd
import dtale
from pandas_profiling import ProfileReport

import config

def run(tool, choose_data):
    if tool == "dtale":
        if choose_data:
            dtale.show(subprocess=False)
        else:
            df = pd.read_csv(config.TRAINING_FILE)
            dtale.show(df, subprocess=False)
    elif tool == "pandas_profiling":
        df = pd.read_csv(config.TRAINING_FILE)
        profile = ProfileReport(df, title="Pandas ProfileReport")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA on the data")
    parser.add_argument("--tool", type=str, help='Select eda tool')    
    parser.add_argument("--choose_data", action="store_true", help='Select data with web interface')

    args = parser.parse_args()
    run(tool=args.tool, choose_data=args.choose_data)

