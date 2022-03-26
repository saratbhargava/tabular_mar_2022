import argparse
import joblib
from pathlib import Path

import config
import pandas as pd
from sklearn import preprocessing

from explainerdashboard import ClassifierExplainer, ExplainerDashboard


def run(fold, num_valid, model_filename):

    # read the data
    df = pd.read_csv(
        f"{config.TRAINING_FILE[:-4]}_folds.csv")

    # set the index
    df = df.set_index("row_id")
    df.index.name = config.INDEX_NAME

    df_valid = df[df['fold'] == fold]
    df_valid = df_valid.drop(["fold"], axis=1)

    df_valid_sample = df_valid.sample(num_valid, random_state=config.RANDOM_STATE)

    y_valid = df_valid_sample[config.TARGET_LABEL]
    X_valid = df_valid_sample.drop(config.TARGET_LABEL, axis=1)
    y_valid.name = config.TARGET_NAME

    print(X_valid.shape, y_valid.shape)

    # Load the model
    model_filepath = Path(config.MODELS) / model_filename
    model_obj = joblib.load(model_filepath)

    # Apply labelencoder
    le = preprocessing.LabelEncoder()
    le.fit(df_valid[config.TARGET_LABEL])

    y_valid = le.transform(y_valid)

    # create an explainer board
    explainer = ClassifierExplainer(
        model_obj, X_valid, y_valid,
        labels=le.classes_)

    db = ExplainerDashboard(explainer, title="Bacteria Explainer",
                            whatif=False, # you can switch off tabs with bools
                            shap_dependence=False,
                            shap_interaction=False,
                            decision_trees=False)
    db.run(port=config.DASHBOARD_PORT)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explain ML model")

    parser.add_argument("--fold", type=int, help='Fold to use for validation')
    parser.add_argument("--num_valid", type=int, help='Number of validation examples')
    parser.add_argument("--model_filename", type=str, help='Model filename')

    args = parser.parse_args()

    run(fold=args.fold, num_valid=args.num_valid, model_filename=args.model_filename)
