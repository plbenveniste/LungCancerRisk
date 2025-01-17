"""
In this file we compute the UPSTF recommendations for the preprocessed NLST and PLCO datasets.
We then compute the precision and recall scores for the UPSTF recommendations.

Args:
    --data-plco: str, path to the preprocessed PLCO data
    --data-nlst: str, path to the preprocessed NLST data

Returns:
    None

Example:
    python UPSTF_recommandations.py --data-plco data/plco_data.csv --data-nlst data/nlst_data.csv

Pierre-Louis Benveniste
"""
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def get_parser():
    """
    This function parses the arguments passed to the script.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='UPSTF recommendations')
    parser.add_argument('--data-plco', type=str, required=True, help='Path to the preprocessed PLCO data')
    parser.add_argument('--data-nlst', type=str, required=True, help='Path to the preprocessed NLST data')
    return parser


def main():
    """
    This function computes the UPSTF recommendations for the preprocessed NLST and PLCO datasets.
    We then compute the precision and recall scores for the UPSTF recommendations.

    Args:
        None

    Returns:
        None
    """
    # Parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    plco = args.data_plco
    nlst = args.data_nlst

    # Load the data
    plco = pd.read_csv(plco)
    nlst = pd.read_csv(nlst)

    # Compute the UPSTF recommendations for PLCO
    print("Pre-processed PLCO size:", len(plco))
    print("Pre-processed PLCO with lung cancer:", len(plco[plco["lung_cancer"]==1]))

    plco_criteria = plco.copy()
    plco_criteria = plco_criteria[plco_criteria["age"]>=50]
    plco_criteria = plco_criteria[plco_criteria["age"]<=80]
    plco_criteria = plco_criteria[plco_criteria["pack_years"]>=20]
    plco_criteria = plco_criteria[ (plco_criteria["cig_stat"]==1) | (plco_criteria["age"] - plco_criteria["ssmokea_f"] <=15) ]

    print("Patients from PLCO who fit into US recommendation:", len(plco_criteria))
    print("Patients from PLCO who fit into US recommendation with lung cancer:", len(plco_criteria[plco_criteria["lung_cancer"]==1]))

    # For each subject in plco_criteria, we set the predicted value to 1 in plco (using the index)
    plco["predicted"] = 0
    for index in plco_criteria.index:
        plco.at[index, "predicted"] = 1
    tn_plco, fp_plco, fn_plco, tp_plco = confusion_matrix(plco["lung_cancer"], plco["predicted"]).ravel()

    print("------- USPSTF RECOMMENDATION ON PLCO --------")
    print("TP : ", tp_plco)
    print("FN : ", fn_plco)
    print("TN : ", tn_plco)
    print("FP : ", fp_plco)
    print("Precision : ", precision_score(plco["lung_cancer"], plco["predicted"]))
    print("Recall : ", recall_score(plco["lung_cancer"], plco["predicted"]))
    print("----------------------------------------------")

    # Compute the UPSTF recommendations for NLST
    print("Pre-processed NLST size:", len(nlst))
    print("Pre-processed NLST with cancer:", len(nlst[nlst["lung_cancer"]==1]))

    nlst_criteria = nlst.copy()
    nlst_criteria = nlst_criteria[nlst_criteria["age"]>=50]
    nlst_criteria = nlst_criteria[nlst_criteria["age"]<=80]
    nlst_criteria = nlst_criteria[nlst_criteria["pack_years"]>=20]
    nlst_criteria = nlst_criteria[ (nlst_criteria["cig_stat"]==1) | (nlst_criteria["age"] - nlst_criteria["ssmokea_f"] <=15) ]

    print("Patients from NLST who fit into US recommendation:", len(nlst_criteria))
    print("Patients from NLST who fit into US recommendation with cancer:", len(nlst_criteria[nlst_criteria["lung_cancer"]==1]))

    # For each subject in nlst_criteria, we set the predicted value to 1 in nlst (using the index)
    nlst["predicted"] = 0
    for index in nlst_criteria.index:
        nlst.at[index, "predicted"] = 1
    tn_nlst, fp_nlst, fn_nlst, tp_nlst = confusion_matrix(nlst["lung_cancer"], nlst["predicted"]).ravel()

    print("------- USPSTF RECOMMENDATION ON NLST --------")
    print("TP : ", tp_nlst)
    print("FN : ", fn_nlst)
    print("TN : ", tn_nlst)
    print("FP : ", fp_nlst)
    print("Precision : ", precision_score(nlst["lung_cancer"], nlst["predicted"]))
    print("Recall : ", recall_score(nlst["lung_cancer"], nlst["predicted"]))
    print("----------------------------------------------")

    return None


if __name__ == '__main__':
    main()