import argparse

import pandas as pd

from sklearn import model_selection

import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Sample train set for testing")

    parser.add_argument("--num_samples", type=int, default=1_000,
                        help='Number of samples')

    args = parser.parse_args()
    
    # Read the input data file
    df = pd.read_csv(config.TRAINING_FILE)

    # Create the kfold column
    df["fold"] = -1
    df = df.sample(args.num_samples).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=config.N_SPLITS)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'fold'] = fold

    # Save the new csv file
    df.to_csv(f"{config.TRAINING_FILE[:-4]}_folds_sample.csv", index=False)
    
