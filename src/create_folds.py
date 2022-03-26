import config

import pandas as pd

from sklearn import model_selection

if __name__ == "__main__":
    # Read the input data file
    df = pd.read_csv(config.TRAINING_FILE)

    # Create the kfold column
    df["fold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    if config.VALIDATION_TYPE == "Kfold":
        kf = model_selection.KFold(n_splits=config.N_SPLITS)
        for fold, (trn_, val_) in enumerate(kf.split(X=df)):
            df.loc[val_, 'fold'] = fold
    elif config.VALIDATION_TYPE == "StratifiedKfold":
        targets = df[config.TARGET_LABEL].values
        kf = model_selection.StratifiedKFold(n_splits=config.N_SPLITS)
        for fold, (trn_, val_) in enumerate(kf.split(X=df, y=targets)):
            df.loc[val_, 'fold'] = fold
    else:
        raise ValueError(f"VALIDATION_TYPE: {config.VALIDATION_TYPE} is invalid")

    # Save the new csv file
    df.to_csv(f"{config.TRAINING_FILE[:-4]}_folds.csv", index=False)
