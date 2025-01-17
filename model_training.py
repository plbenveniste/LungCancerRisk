""""
In this file we perform feature selection and model training.
The model trained is an XGBoost model.

Args: 
    --data_plco: str, path to the preprocessed training data
    --data_nlst: str, path to the preprocessed test data
    --output_path: str, path to save the model

Returns:
    None

Example:
    python model_comparison.py --data-plco data/plco_data.csv --data-nlst data/nlst_data.csv --output-path models/

Pierre-Louis Benveniste
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
# import lightgbm as lgb
# from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pickle
import shap
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def get_parser():
    """
    This function parses the arguments passed to the script.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Model comparison')
    parser.add_argument('--data-plco', type=str, required=True, help='Path to the preprocessed PLCO data')
    parser.add_argument('--data-nlst', type=str, required=True, help='Path to the preprocessed NLST data')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the model')
    return parser


def main():
    """
    This function performs model training of the XGB model.

    Args:
        None

    Returns:
        None
    """
    args = get_parser()
    args = args.parse_args()

    plco = pd.read_csv(args.data_plco)
    nlst = pd.read_csv(args.data_nlst)

    # we remove height and weight since they are correlated with BMI
    plco = plco.drop(columns=['height_f', 'weight_f'])
    nlst = nlst.drop(columns=['height_f', 'weight_f'])

    # Split the data into features and target variable
    x_plco = plco.drop(columns= ['lung_cancer'])
    y_plco = plco['lung_cancer']
    x_nlst = nlst.drop(columns=['lung_cancer'])
    y_nlst = nlst['lung_cancer']

    x_train, x_val, y_train, y_val = train_test_split(x_plco, y_plco, test_size=0.3, random_state=0)
    x_test, y_test = x_nlst, y_nlst

    #------------------------ Initial model training ------------------------
    ## We hide the Bayesian search since we now have the best parameters
    # search_spaces = {
    #     'learning_rate': Real(0.001, 0.5),
    #     'n_estimators': Integer(10, 1000),
    #     'max_depth': Integer(3, 10),
    #     'min_child_weight': Integer(1, 10),
    #     'subsample': Real(0.1, 1),
    #     'colsample_bytree': Real(0.001, 1),
    #     'gamma': Real(0, 1),
    #     'reg_alpha': Real(0, 1),
    #     'reg_lambda': Real(0, 1),
    # }
    # # We define the model
    # model = XGBClassifier(seed=42,)
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # # We fit the search
    # search.fit(x_train, y_train)

    ## Model training
    model = XGBClassifier(colsample_bytree=0.001, gamma=1.0, learning_rate=0.1202500016477047, max_depth=3, min_child_weight=4,
                           n_estimators=360, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0, seed=42)
    model.fit(x_train, y_train)

    # Prediction on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute performance
    print("Model performance after hyperparameter tuning (fewer features with 3 year deadline)")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    # Print confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\n")

    #------------------------ Feature selection ------------------------
    # We select the 10 most contributing features
    #To see importance over the entire dataset
    # shap_values = shap.Explainer(model).shap_values(x_train)
    # shap.summary_plot(shap_values, x_train, plot_type="bar")

    # The most contributing features are:
    top_8_features = ['cig_years', 'age', 'pack_years', 'ssmokea_f', 'cig_stat', 'cigpd_f', 'bmi', 'lung_fh']

    # We keep only these features
    x_train = x_train[top_8_features]
    x_val = x_val[top_8_features]
    x_test = x_test[top_8_features]

    search_spaces = {
        'learning_rate': Real(0.001, 0.5),
        'n_estimators': Integer(10, 1000),
        'max_depth': Integer(3, 10),
        'min_child_weight': Integer(1, 10),
        'subsample': Real(0.1, 1),
        'colsample_bytree': Real(0.001, 1),
        'gamma': Real(0, 1),
        'reg_alpha': Real(0, 1),
        'reg_lambda': Real(0, 1),
    }
    # We define the model
    model = XGBClassifier(seed=42,)
    # We define the search
    search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # We fit the search
    search.fit(x_train, y_train)
    model = search.best_estimator_
    # We print the best parameters
    print(search.best_params_)

    # Prediction on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute performance
    print("Model performance after hyperparameter tuning (fewer features with 3 year deadline)")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    # Print confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\n")

    ############### CALIBRATION OF THE MODEL     ############################
    # We calibrate the model using the isotonic method
    model_calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    model_calibrated.fit(x_train, y_train)

    # Performance on the test set
    y_test_pred_calibrated = model_calibrated.predict(x_test)
    y_test_proba_calibrated = model_calibrated.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Final model performance on the testing set after calibration")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba_calibrated))
    print("Brier score:", brier_score_loss(y_test, y_test_proba_calibrated))
    print("Average precision:", precision_score(y_test, y_test_pred_calibrated))
    print("Average Recall:", recall_score(y_test, y_test_pred_calibrated))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred_calibrated))
    precision_test_calibrated, recall_test_calibrated, _ = precision_recall_curve(y_test, y_test_pred_calibrated)
    print("AUC-PR score:", auc(recall_test_calibrated, precision_test_calibrated))
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_calibrated).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")

    # Plot the calibration before and after calibration
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_test_proba_calibrated, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated')
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve of the final model')
    plt.legend()
    plt.show()

    # Plot the calibration curve only of the calibrated model
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_test_proba_calibrated, n_bins=10)
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='Calibrated XGBoost model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve of the final model')
    plt.legend()
    plt.show()

    return None


if __name__ == '__main__':
    main()