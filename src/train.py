import argparse
from datetime import datetime
from pathlib import Path

import joblib
import rich
import wandb

import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn import metrics, model_selection, pipeline, preprocessing

import config
import feature_extraction
import model_dispatcher


def run(fold, model, features, tune, num_trails, model_filename):
    
    # read the data
    df = pd.read_csv( 
        f"{config.TRAINING_FILE[:-4]}_folds.csv")
    df = df.set_index("row_id")
    df.index.name = config.INDEX_NAME

    if fold == -1:
        y = df[config.TARGET_LABEL]
        X = df.drop(["fold", config.TARGET_LABEL], axis=1)

        # Apply labelencoder
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        feature_names = list(X.columns)

    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

        df_train = df_train.drop(["fold",], axis=1)
        df_valid = df_valid.drop(["fold",], axis=1)

        # Create train features and target labels
        y_train = df_train[config.TARGET_LABEL]
        X_train = df_train.drop(config.TARGET_LABEL, axis=1)

        y_valid = df_valid[config.TARGET_LABEL]
        X_valid = df_valid.drop(config.TARGET_LABEL, axis=1)

        feature_names = list(X_train.columns)

        # Apply labelencoder
        le = preprocessing.LabelEncoder()
        le.fit(y_train)        
        y_train = le.transform(y_train)
        y_valid = le.transform(y_valid)


    def get_model(hyper_param_dict):
        return pipeline.make_pipeline(
            *feature_extraction.features[features],
            model_dispatcher.models[model](**hyper_param_dict))

    # hyper params optimization
    def objective(hyper_param_dict):
        model_obj = get_model(hyper_param_dict)
        model_obj.fit(X_train, y_train)
        acc = model_obj.score(X_valid, y_valid)
        train_acc = model_obj.score(X_train, y_train)
        return {"loss": -acc, "status": STATUS_OK,
                "train_acc": train_acc,
                "model": model_obj,
                "hyper_param_dict": hyper_param_dict}


    def k_fold_objective(hyper_param_dict):
        model_obj = get_model(hyper_param_dict)
        cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True,
                                             random_state=config.RANDOM_STATE)
        cv_scores = model_selection.cross_val_score(
            estimator=model_obj, X=X, y=y,
            cv=cv, n_jobs=-1)
        model_obj.fit(X, y)
        train_acc = model_obj.score(X_train, y_train)
        return {"loss": -np.mean(cv_scores), "train_acc": train_acc, "status": STATUS_OK,
                "model": model_obj, "hyper_param_dict": hyper_param_dict}


    def track_model_with_wandb(trained_model, hyper_param_dict, train_acc,
                               valid_acc, wandb_plot_classifier,
                               model_filename=""):
        run = wandb.init(project=config.PROJECT, entity=config.ENTITY, reinit=True)
        wandb.config.update(hyper_param_dict)
        wandb.log({'valid/accuracy': valid_acc, 'train/accuracy': train_acc,
                   "model_filename": model_filename})
        if wandb_plot_classifier:
            # visualize the model performance
            y_pred = trained_model.predict(X_valid)
            y_pred_probas = trained_model.predict_proba(X_valid)
            wandb.sklearn.plot_classifier(
                trained_model,
                X_train, X_valid, y_train, y_valid,
                y_pred, y_pred_probas, np.unique(y_train),
                model_name=model, feature_names=feature_names)
        run.finish()


    if fold == -1:
        objective_fn = k_fold_objective
    else:
        objective_fn = objective

    if tune:
        trials = Trials()
        best = fmin(
            fn = objective_fn,
            space = config.hyper_params[model],
            algo = tpe.suggest,
            max_evals = num_trails,
            trials = trials,
        )

        min_trail_idx = np.argmin([result['loss'] for result in trials.results])
        for idx, result in enumerate(trials.results):
            track_model_with_wandb(result['model'], result['hyper_param_dict'],
                                   train_acc=result['train_acc'],
                                   valid_acc=-result['loss'],
                                   wandb_plot_classifier=(fold>=0),
                                   model_filename=model_filename if idx==min_trail_idx else "")

        # save the best model
        best_model = trials.results[min_trail_idx]['model']
    else:
        obj_dict = objective_fn(config.fixed_hyper_params[model])
        best_model = obj_dict['model']
        track_model_with_wandb(best_model, obj_dict['hyper_param_dict'],
                               train_acc=obj_dict['train_acc'],
                               valid_acc=-obj_dict['loss'],
                               wandb_plot_classifier=True,
                               model_filename=model_filename)
        rich.print(obj_dict['hyper_param_dict'])
        rich.print(f"train_acc: {obj_dict['train_acc']}, valid_acc: {-obj_dict['loss']}")

    joblib.dump(
        best_model,
        Path(config.MODELS) / model_filename
    )
    
    return
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")

    parser.add_argument("--fold", type=int, help='Fold to use for validation')
    parser.add_argument("--model", type=str, help='ML model',
                        choices=model_dispatcher.models.keys())
    parser.add_argument("--features", type=str,
                        help='Feature engineering',
                        default="",
                        choices=feature_extraction.features.keys())
    parser.add_argument("--tune", action="store_true", help='Tune model parameters')
    parser.add_argument("--num_trails", type=int,
                        help='Number of trials for hyperparam tuning', default=3)
    parser.add_argument("--model_filename", type=str, help='Model filename')

    args = parser.parse_args()

    run(fold=args.fold, model=args.model, features=args.features,
        tune=args.tune, num_trails=args.num_trails,
        model_filename=args.model_filename)
