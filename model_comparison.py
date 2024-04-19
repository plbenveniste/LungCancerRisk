""""
In this file we perform model training or model evaluation on the preprocessed data.
We compare the performance of the LGB and the XGB models.

Args: 
    --data_plco: str, path to the preprocessed training data
    --data_nlst: str, path to the preprocessed test data
    --mode: str, 'train' or 'eval'
    --output_path: str, path to save the model

Returns:
    None

Example:
    python model_comparison.py --data_plco data/plco_data.csv --data_nlst data/nlst_data.csv --mode train --output_path models/

Pierre-Louis Benveniste
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
import lightgbm as lgb
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score


def get_parser():
    """
    This function parses the arguments passed to the script.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Model comparison')
    parser.add_argument('--data_plco', type=str, required=True, help='Path to the preprocessed PLCO data')
    parser.add_argument('--data_nlst', type=str, required=True, help='Path to the preprocessed NLST data')
    parser.add_argument('--mode', help='Mode: train or eval', choices=['train', 'eval'])
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the model')
    return parser


def train_model(x_train, x_val, y_train, y_val, output_path, features):
    """
    This function trains the LGB and XGB models and saves the best model.

    Args:
        x_train: pd.DataFrame, features of the training set
        x_val: pd.DataFrame, features of the validation set
        y_train: pd.Series, target variable of the training set
        y_val: pd.Series, target variable of the validation set
        output_path: str, path to save the model
    
    Returns:
        xgb_model: XGB model
        lgb_model: LGB model
    """
    #------------------------ Train the LGB model ------------------------
        # Define the parameter search space for Bayesian optimization
    param_space_lgb = {
        'num_leaves': Integer(10, 100),
        'max_depth': Integer(3, 12),
        'n_estimators': Integer(100, 500),
        'learning_rate': Real(0.001, 0.1, 'log-uniform'),
        'min_child_samples': Integer(10, 100),
        'subsample': Real(0.1, 1.0, 'uniform'),
        'colsample_bytree': Real(0.1, 1.0, 'uniform'),
        'reg_alpha': Real(0.001, 1.0, 'uniform'),
        'reg_lambda': Real(0.001, 1.0, 'uniform'),
    }

    # Fixed parameters for the model
    fixed_params_lgb = {
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 0,
        'boosting_type': 'gbdt',
        'is_unbalance': True
    }

    # Initialize the LGBMClassifier
    lgb_model = LGBMClassifier(**fixed_params_lgb)

    # Perform Bayesian hyperparameter optimization
    bayes_search_lgb = BayesSearchCV(estimator=lgb_model, search_spaces=param_space_lgb, cv=5)
    bayes_search_lgb.fit(x_train, y_train)

    # Print the best hyperparameters and the corresponding score
    print("----------------- LightGBM model -----------------")
    print("Best Hyperparameters (Bayesian): ", bayes_search_lgb.best_params_)
    print("Best Score (Bayesian): ", bayes_search_lgb.best_score_)

    # Save the best model
    
    lgb_model = bayes_search_lgb.best_estimator_
    lgb_model.booster_.save_model(output_path + '/lgb_model' + features + '.txt')

    #------------------------ Train the XGB model ------------------------
    # Initialize the XGBClassifier fixed parameters
    fixed_params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'exact',
        'random_state': 0
    }
    search_space_xgb = {
        'learning_rate': Real(0.01, 1.0, 'uniform'),
        'max_depth': Integer(2, 10),
        'subsample': Real(0.1, 1.0, 'uniform'),
        'colsample_bytree': Real(0.1, 1.0, 'uniform'),
        'n_estimators': Integer(50, 1000),
        'reg_alpha': Real(0.001, 100., 'uniform'), 
       }

    xgb_model = XGBClassifier(**fixed_params_xgb)
    # Perform Bayesian hyperparameter optimization
    bayes_search_xgb = BayesSearchCV(estimator=xgb_model, search_spaces=search_space_xgb, cv=5)
    bayes_search_xgb.fit(x_train, y_train)

    # Print the best hyperparameters and the corresponding score
    print("----------------- XGBoost model -----------------")
    print("Best Hyperparameters (Bayesian - XGBoost): ", bayes_search_xgb.best_params_)
    print("Best Score (Bayesian - XGBoost): ", bayes_search_xgb.best_score_)

    # Save the best model
    xgb_model = bayes_search_xgb.best_estimator_
    xgb_model.save_model(output_path + '/xgb_model' + features + '.txt')

    return lgb_model, xgb_model


def main():
    """
    This function performs model training or model evaluation on the preprocessed data.
    We compare the performance of the LGB and the XGB models.

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

    if args.mode == 'train':
        # Split the data into training and testing sets
        x_train, x_val, y_train, y_val = train_test_split(x_plco, y_plco, test_size=0.3, random_state=0)
        # We first train the models on all the features
        lgb_all, xgb_all = train_model(x_train, x_val, y_train, y_val, args.output_path, '_all_features')

        # We extract the features importance
        columns_to_keep = ['age', 'ssmokea_f', 'cig_stat', 'pack_years', 'smokea_f', 
                        'cig_years', 'lung_fh', 'bmi'] 
        x_plco = x_plco[columns_to_keep]

        # Split the data into training and testing sets
        x_train, x_val, y_train, y_val = train_test_split(x_plco, y_plco, test_size=0.3, random_state=0)

        # We then train the models on the selected features
        lgb_final, xgb_final = train_model(x_train, x_val, y_train, y_val, args.output_path, '_final_features')
    
    elif args.mode == 'eval':
        # Split the data into training and testing sets
        x_train, x_val, y_train, y_val = train_test_split(x_plco, y_plco, test_size=0.3, random_state=0)

        # The models are trained using the best hyperparameters found in previous work
        colsample_bytree = 0.37747061190521813
        learning_rate = 0.001524291252423568
        max_depth = 12
        min_child_samples = 10
        n_estimators = 500
        num_leaves = 100
        reg_alpha = 0.01991764226751736
        reg_lambda = 0.980934774946671
        subsample = 0.1609835879068368
        lgb_all = LGBMClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_samples=min_child_samples,
                                    n_estimators=n_estimators, num_leaves=num_leaves, reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample=subsample, 
                                    objective='binary', metric='auc', random_state=0, boosting_type='gbdt', is_unbalance=True)
        lgb_all.fit(x_train, y_train)

        # Now for the XGB model
        colsample_bytree = 0.9757843693897756
        learning_rate = 0.06357032714696428
        max_depth = 10
        n_estimators = 329
        reg_alpha = 36.731037664039135
        subsample = 0.8382196467440679
        xgb_all = XGBClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                reg_alpha=reg_alpha, subsample=subsample, objective='binary:logistic', eval_metric='aucpr', tree_method='exact', random_state=0)

        # Fit the model on the training data
        xgb_all.fit(x_train, y_train)

        # we remove some of the features
        columns_to_keep = ['age', 'ssmokea_f', 'cig_stat', 'pack_years', 'smokea_f', 
                        'cig_years', 'lung_fh', 'bmi'] 
        x_plco = x_plco[columns_to_keep]

        # Split the data into training and testing sets
        x_train, x_val, y_train, y_val = train_test_split(x_plco, y_plco, test_size=0.3, random_state=0)

        # The models are trained using the best hyperparameters found in previous work
        colsample_bytree = 0.3898449137185822
        learning_rate = 0.0020319229167561803
        max_depth = 6
        min_child_samples = 12
        n_estimators = 373
        num_leaves = 21
        reg_alpha = 0.856025438690339
        reg_lambda = 0.9206453463657929
        subsample = 0.8542248250324743
        lgb_final = LGBMClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_samples=min_child_samples,
                                    n_estimators=n_estimators, num_leaves=num_leaves, reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample=subsample, 
                                    objective='binary', metric='auc', random_state=0, boosting_type='gbdt', is_unbalance=True)
        lgb_final.fit(x_train, y_train)

        # Now for the XGB model
        colsample_bytree = 0.5215579121066478
        learning_rate = 0.9674062745251472
        max_depth = 3
        n_estimators = 619
        reg_alpha = 35.683424042437835
        subsample = 0.2798727981229301
        xgb_final = XGBClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                reg_alpha=reg_alpha, subsample=subsample, objective='binary:logistic', eval_metric='aucpr', tree_method='exact', random_state=0)

        # Fit the model on the training data
        xgb_final.fit(x_train, y_train)
    
    # Now we evaluate the models on the NLST data
    x_test, y_test = x_nlst, y_nlst

    # We first evaluate the models on all the features
    # Evaluate the LGB model on the test set
    y_pred_lgb = lgb_all.predict(x_test)
    y_pred_proba_lgb = lgb_all.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    roc_auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)
    accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
    precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_pred_lgb)
    brier_score_lgb = brier_score_loss(y_test, y_pred_proba_lgb)
    pr_auc_lgb = auc(recall_lgb, precision_lgb)

    print("SCORES FOR LGB MODEL ON ALL FEATURES")
    print("ROC AUC Score: ", roc_auc_lgb)
    print("Accuracy Score: ", accuracy_lgb)
    print("Brier score ", brier_score_lgb )
    print("AUC-PR score  ", pr_auc_lgb)

    # Evaluate the XGB model on the test set
    y_pred_xgb = xgb_all.predict(x_test)
    y_pred_proba_xgb = xgb_all.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_xgb)
    brier_score_xgb = brier_score_loss(y_test, y_pred_proba_xgb)
    pr_auc_xgb = auc(recall_xgb, precision_xgb)

    print("SCORES FOR XGB MODEL ON ALL FEATURES")
    print("ROC AUC Score: ", roc_auc_xgb)
    print("Accuracy Score: ", accuracy_xgb)
    print("Brier score ", brier_score_xgb)
    print("AUC-PR score  ", pr_auc_xgb)

    ############################################################
    # We then evaluate the models on the selected features
    columns_to_keep = ['age', 'ssmokea_f', 'cig_stat', 'pack_years', 'smokea_f', 
                        'cig_years', 'lung_fh', 'bmi'] 
    x_plco = x_plco[columns_to_keep]
    x_nlst = x_nlst[columns_to_keep]
    x_test, y_test = x_nlst, y_nlst
    # Evaluate the LGB model on the test set
    y_pred_lgb = lgb_final.predict(x_test)
    y_pred_proba_lgb = lgb_final.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    roc_auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)
    accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
    precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_pred_lgb)
    brier_score_lgb = brier_score_loss(y_test, y_pred_proba_lgb)
    pr_auc_lgb = auc(recall_lgb, precision_lgb)

    print("SCORES FOR LGB MODEL ON FINAL FEATURES")
    print("ROC AUC Score: ", roc_auc_lgb)
    print("Accuracy Score: ", accuracy_lgb)
    print("Brier score ", brier_score_lgb )
    print("AUC-PR score  ", pr_auc_lgb)

    # We extract the precision for a fixed recall on the plco dataset
    y_pred_proba_lgb = lgb_final.predict_proba(x_plco)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_plco, y_pred_proba_lgb)
    recall_value_plco = 0.765
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_plco = df.loc[df['recall'] >= recall_value_plco].precision.max() 
    print("On PLCO : For recall = " + str(round(recall_value_plco,3))  + " precision is : " + str(round(max_precision_plco,3)))
    
    y_pred_proba_lgb = lgb_final.predict_proba(x_nlst)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_nlst, y_pred_proba_lgb)
    recall_value_nlst = 0.989
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_nlst = df.loc[df['recall'] >= recall_value_nlst].precision.max() 
    print("On NLST : For recall = " + str(round(recall_value_nlst,3))  + " precision is : " + str(round(max_precision_nlst,3)))

    # Evaluate the XGB model on the test set
    y_pred_xgb = xgb_final.predict(x_test)
    y_pred_proba_xgb = xgb_final.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_xgb)
    brier_score_xgb = brier_score_loss(y_test, y_pred_proba_xgb)
    pr_auc_xgb = auc(recall_xgb, precision_xgb)

    print("SCORES FOR XGB MODEL ON FINAL FEATURES")
    print("ROC AUC Score: ", roc_auc_xgb)
    print("Accuracy Score: ", accuracy_xgb)
    print("Brier score ", brier_score_xgb)
    print("AUC-PR score  ", pr_auc_xgb)

    # We extract the precision for a fixed recall on the plco dataset
    y_pred_proba_xgb = xgb_final.predict_proba(x_plco)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_plco, y_pred_proba_xgb)
    recall_value_plco = 0.765
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_plco = df.loc[df['recall'] >= recall_value_plco].precision.max() 
    print("For recall = " + str(round(recall_value_plco,3))  + " precision is : " + str(round(max_precision_plco,3)) )

    # We extract the precision for a fixed recall on the plco dataset
    y_pred_proba_lgb = xgb_final.predict_proba(x_plco)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_plco, y_pred_proba_lgb)
    recall_value_plco = 0.765
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_plco = df.loc[df['recall'] >= recall_value_plco].precision.max() 
    print("On PLCO : For recall = " + str(round(recall_value_plco,3))  + " precision is : " + str(round(max_precision_plco,3)))
    
    y_pred_proba_lgb = xgb_final.predict_proba(x_nlst)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_nlst, y_pred_proba_lgb)
    recall_value_nlst = 0.989
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_nlst = df.loc[df['recall'] >= recall_value_nlst].precision.max() 
    print("On NLST : For recall = " + str(round(recall_value_nlst,3))  + " precision is : " + str(round(max_precision_nlst,3)))

    return None


if __name__ == '__main__':
    main()