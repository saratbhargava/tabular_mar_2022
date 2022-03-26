import argparse
import joblib
from pathlib import Path

import pandas as pd

import config

from sklearn.preprocessing import LabelEncoder


def run(model_filenames, ensemble, submit_filename):

    # Load the testing data
    df = pd.read_csv(config.TESTING_FILE)
    row_ids = df['row_id']
    df = df.drop("row_id", axis=1)

    X_test = df.values

    # Load the training data
    df_train = pd.read_csv( 
        f"{config.TRAINING_FILE[:-4]}_folds.csv")
    y_train = df_train[config.TARGET_LABEL]
    
    # Apply labelencoder
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    if ensemble == "":
        # Load the model
        model_filename = model_filenames[0]
        model_filepath = Path(config.MODELS) / model_filename
        model_obj = joblib.load(model_filepath)

        # Predict the X_test
        y_test_pred = model_obj.predict(X_test)
        y_test_pred = le.inverse_transform(y_test_pred)
        y_test_pred = pd.Series(y_test_pred, index=row_ids,
                                name=config.TARGET_LABEL)
    elif ensemble == "mean_predictions":
        y_pred_probas_list = []
        # Load the models        
        for model_filename in model_filenames:
            model_filepath = Path(config.MODELS) / model_filename
            model_obj = joblib.load(model_filepath)

            # Predict the X_test
            y_test_pred_proba = model_obj.predict_proba(X_test)
            y_test_pred_proba_list.append(y_test_pred_proba)
        y_test_pred_proba_ensemble = np.mean(y_test_pred_proba_list, axis=0)
        y_test_pred_ensemble = np.argmax(y_test_pred_proba_ensemble, axis=1)
        y_test_pred = le.inverse_transform(y_test_pred_ensemble)
        y_test_pred = pd.Series(y_test_pred, index=row_ids,
                                name=config.TARGET_LABEL)
    elif ensemble == "max_voting":
        # y_pred_probas_list = []
        # # Load the models        
        # for model_filename in model_filenames:
        #     model_filepath = Path(config.MODELS) / model_filename
        #     model_obj = joblib.load(model_filepath)

        #     # Predict the X_test
        #     y_test_pred = model_obj.predict(X_test)
        #     y_test_pred_list.append(y_test_pred_proba)
        # y_test_pred_proba_ensemble = np.mean(y_test_pred_proba_list, axis=0)
        raise NotImplementedError(f"{ensemble=} is not defined")
    
    # Save the predictions
    submit_filepath = Path(config.SUBMIT) / submit_filename
    y_test_pred.to_csv(submit_filepath)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_filenames", nargs="+", type=list)
    parser.add_argument("--submit_filename", type=str)
    parser.add_argument("--ensemble", type=str,
                        choices=["", "mean_predictions", "max_voting"],
                        default="")

    args = parser.parse_args()

    run(model_filename=args.model_filenames,
        submit_filename=args.submit_filename)
