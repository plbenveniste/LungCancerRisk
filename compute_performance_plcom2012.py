"""
In this script we look at the performance of the PLCOM2012 model on the NLST and PLCO datasets.
To compute the performance, we will use the following script: https://github.com/resplab/PLCOm2012/blob/main/R/plcom2012.R

Args:
    --nlst-path: Path to the preprocessed NLST data
    --plco-path: Path to the preprocessed PLCO data

Returns:
    None

Example:
    python compute_performance_plcom2012.py --nlst-path data/NLST.csv --plco-path data/PLCO.csv

Pierre-Louis Benveniste
"""

import argparse
import pandas as pd
import math


def get_parser():
    """
    This function parses the arguments passed to the script.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Compute Performance of PLCOM2012')
    parser.add_argument('--nlst-path', type=str, help='Path to the preprocessed NLST data')
    parser.add_argument('--plco-path', type=str, help='Path to the preprocessed PLCO data')
    return parser


def model_plcom2012(age, race, education, bmi, copd, cancer_hist, family_hist_lung_cancer, smoking_status, smoking_intensity, duration_smoking,
                      smoking_quit_time):
    """
    This function computes the PLCOM2012 model.

    Args:
        age: a vector of patient's age
        race: categorical variable of patient's race or ethnic group (White, Black, Hispanic, Asian, American Indian, Alaskan Native, Native Hawaiian, Pacific Islander)
        education: education was measured in six ordinal levels: less than high-school graduate (level 1), high-school graduate (level 2), some training after high school (level 3), some college (level 4), 
                    college graduate (level 5), and postgraduate or professional degree (level 6)
        bmi: a vector of patient's body mass index, per 1 unit of increase
        copd: binary variable of chronic obstructive pulmonary disease (yes as 1 or no as 0)
        cancer_hist: binary variable of patient's cancer history (yes as 1 or no as 0)
        family_hist_lung_cancer: binary variable of patient's family history of lung cancer (yes as 1 or no as 0)
        smoking_status: binary variable of patient's smoking status (current as 1 or former as 0)
        smoking_intensity: a vector of the number cigarettes patient smokes per day
        duration_smoking: a vector of patient's duration of smoking, per 1-yr increase
        smoking_quit_time: a vector of patient's smoking quit time, per 1-yr increase
    
    Returns:
        risk: a vector of patient's risk of lung cancer in the next 6 years
    """
    if race in ["white", "american indian", "alaskan native", 1]:
        model = 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)^(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506

    if race in ["black", 2]:
        model = 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)^(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 + 0.3944778

    if race in ["hispanic", 3]:
        model <- 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)^(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 - 0.7434744

    if race in ["asian", 4]:
        model <- 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)^(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 - 0.466585

    if race in ["native hawaiian", "pacific islander", 5]:
        model <- 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)^(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 + 1.027152

    prob = math.exp(model) / (1 + math.exp(model))
    return prob




def main():
    """
    This function computes the performance of the PLCOM2012 model on the NLST and PLCO datasets.

    Args:
        None

    Returns:
        None
    """
    # Parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    nlst_path = args.nlst_path
    plco_path = args.plco_path

    # Loading of both datasets
    plco = pd.read_csv(plco_path)
    nlst = pd.read_csv(nlst_path)

    # We first uniformise both dataset
    # Uniformisation of both datasets
    plco = plco[["age", "race7", "educat", "weight_f", "height_f", "d_seer_death", "ph_first_cancer"
                 "sex", "height_f", "weight_f", "race7", "ssmokea_f", "cig_stat", "cigar", "pipe", "pack_years", "smokea_f", "cigpd_f","cig_years", "bronchit_f",
                    "diabetes_f", "emphys_f", "hearta_f", "hyperten_f", "stroke_f", "lung_fh", "d_seer_death", "lung_cancer",]]
    # For race : 
    plco["race7"] = plco["race7"].replace([6],[5])
    # Remove participant who have race=7
    plco = plco.loc[plco['race7']!=7]
    
    # For education
    plco["educat"] = plco["educat"].replace([3,4,5,6,7],[2,3,4,5,7])
    # Remove participant who have education which is not in [1,2,3,4,5,6]
    plco = plco.loc[plco['educat'].isin([1, 2, 3, 4, 5, 6])]

    # For bmi : round it to the nearest integer
    plco.loc[:, 'bmi'] = plco['weight_f'] / plco['height_f']**2 * 703
    plco.loc[:, 'bmi'] = plco['bmi'].round(0)
    # Remove participant who have no bmi
    plco = plco.loc[plco['bmi'].notnull()]

    # For copd: create a column with binary values : 1 if d_seer_death==50130 and 0 otherwise
    plco['copd'] = 1 * (plco['d_seer_death'] == 50130)
    # Remove participant who have no copd
    plco = plco.loc[plco['copd'].notnull()]

    # For cancer_hist: create a column with binary values : 1 if ph_first_cancer is a number and 0 otherwise
    plco['cancer_hist'] = 1 * (plco['ph_first_cancer'].notnull())
    # Remove participant who have no cancer_hist
    plco = plco.loc[plco['cancer_hist'].notnull()]

    # For family_hist_lung_cancer: remove participant who have value 9 or not a number
    