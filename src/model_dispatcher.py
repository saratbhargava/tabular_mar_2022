import config

from autosklearn import classification

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

from sklearn import ensemble, tree, svm, preprocessing, pipeline

from xgboost import XGBClassifier

# Declare the various models
models = {
    "decision_tree": tree.DecisionTreeClassifier,
    "rf": ensemble.RandomForestClassifier,
    "xgb": XGBClassifier,
    "svm": svm.LinearSVC,
    "autosklearn": classification.AutoSklearnClassifier,
}


