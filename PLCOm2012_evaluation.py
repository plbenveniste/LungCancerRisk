"""
In this script we look at the performance of the PLCOM2012 model on the NLST and PLCO datasets.
To compute the performance, we will use the following script: https://github.com/resplab/PLCOm2012/blob/main/R/plcom2012.R

Args:
    --nlst: Path to the preprocessed NLST data
    --plco: Path to the preprocessed PLCO data
    --model: Path to the XGB model
    --nlst-preprocessed: Path to the preprocessed NLST data
    --plco-preprocessed: Path to the preprocessed PLCO data

Returns:
    None

Example:
    python PLCOm2012_evaluation.py --nlst-path data/NLST.csv --plco-path data/PLCO.csv --model-path

Pierre-Louis Benveniste
"""

import argparse
import pandas as pd
import math
from sklearn.metrics import precision_recall_curve
import pickle


def get_parser():
    """
    This function parses the arguments passed to the script.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Compute Performance of PLCOM2012')
    parser.add_argument('--nlst', type=str, help='Path to the non-preprocessed NLST data', required=True)
    parser.add_argument('--plco', type=str, help='Path to the non-preprocessed PLCO data', required=True)
    parser.add_argument('--model', type=str, help='Path to the XGB model', required=True)
    parser.add_argument('--nlst-preprocessed', type=str, help='Path to the preprocessed NLST data', required=True)
    parser.add_argument('--plco-preprocessed', type=str, help='Path to the preprocessed PLCO data', required=True)
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
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)**(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506

    if race in ["black", 2]:
        model = 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)**(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 + 0.3944778

    if race in ["hispanic", 3]:
        model = 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)**(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 - 0.7434744

    if race in ["asian", 4]:
        model = 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)**(-1) - 0.4021541613) + 0.0317321 * \
        (duration_smoking - 27) - 0.0308572 * (smoking_quit_time - 10) - 4.532506 - 0.466585

    if race in ["native hawaiian", "pacific islander", 5]:
        model = 0.0778868 * (age - 62) - 0.0812744 * (education - 4) - 0.0274194 * (bmi - 27) + 0.3553063 * copd + 0.4589971 * cancer_hist + \
        0.587185 * family_hist_lung_cancer + 0.2597431 * smoking_status - 1.822606 * ((smoking_intensity/10)**(-1) - 0.4021541613) + 0.0317321 * \
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
    nlst_path = args.nlst
    plco_path = args.plco
    xgb_path = args.model

    ###################### COMPARISON ON PLCO ######################
    print("-- Filtering of PLCO based on PLCOm2012 --")
    plco = pd.read_csv(plco_path)
    print("Number of participants in PLCO for PLCOm2012: " + str(len(plco)))

    # Designed for patients who have smoked or are current smokers
    plco = plco.loc[plco.cig_stat > 0]
    print("Number of participants in PLCO for PLCOm2012 after removing non-smokers: " + str(len(plco)))

    # Uniformisation of PLCO
    plco = plco[["age", "race7", "educat", "weight_f", "height_f", "d_seer_death", "ph_first_cancer",  "lung_fh", "cig_stat",
                  "cigpd_f", "cig_years", "cig_stop", "plco_id", "lung_cancer", "lung_exitdays"]]

    # For age: remove participants which have a non numeric age
    plco = plco.loc[plco['age'].notnull()]

    # For race : 
    plco["race7"] = plco["race7"].replace([6],[1])
    # Remove participant who have race=7 (missing field)
    plco = plco.loc[plco['race7']!=7]
    
    # For education
    plco["educat"] = plco["educat"].replace([2,3,4,5,6,7],[1,2,3,4,5,6])
    # Remove participant who have education which is not in [1,2,3,4,5,6]
    plco = plco.loc[plco['educat'].isin([1, 2, 3, 4, 5, 6])]
    # Remove lines where education is missing
    plco = plco.loc[plco['educat'].notnull()]

    # For bmi : round it to the nearest integer
    # Remove participant who have no weight or height
    plco = plco.loc[plco['weight_f'].notnull()]
    plco = plco.loc[plco['height_f'].notnull()]
    plco.loc[:, 'bmi'] =  703 * plco['weight_f'] / (plco['height_f']**2)
    plco.loc[:, 'bmi'] = plco['bmi'].round(0)
    # Remove participant who have no bmi
    plco = plco.loc[plco['bmi'].notnull()]
    # Remove weight_f and height_f columns
    plco = plco.drop(columns=['weight_f', 'height_f'])

    # For copd: create a column with binary values : 1 if d_seer_death==50130 and 0 otherwise
    plco['copd'] = 1 * (plco['d_seer_death'] == 50130)
    # Remove participant who have no copd
    plco = plco.loc[plco['copd'].notnull()]
    # Remove d_seer_death column
    plco = plco.drop(columns=['d_seer_death'])

    # For cancer_hist: create a column with binary values : 1 if ph_first_cancer is a number and 0 otherwise
    plco['cancer_hist'] = 1 * (plco['ph_first_cancer'].notnull())
    # Remove participant who have no cancer_hist
    plco = plco.loc[plco['cancer_hist'].notnull()]
    # Remove ph_first_cancer column
    plco = plco.drop(columns=['ph_first_cancer'])

    # For family_hist_lung_cancer: remove participant who have value 9 or not a number (ie convert 9 to nan and then remove non number)
    plco['lung_fh'] = plco['lung_fh'].replace([9], None)
    plco = plco.loc[plco['lung_fh'].notnull()]

    # For smoking_status: create a column with binary values : 1 if cig_stat==1 and 0 otherwise. Then remove participant who have no smoking status
    plco['smoking_status'] = 1 * (plco['cig_stat'] == 1)
    plco = plco.loc[plco['smoking_status'].notnull()]
    # Remove cig_stat column
    plco = plco.drop(columns=['cig_stat'])

    # For smoking_intensity: create a column with the number of cigarettes smoked per day. Then remove participant who have no smoking intensity
    plco['smoking_intensity'] = plco['cigpd_f'].replace([1,2,3,4,5,6,7], [5,15,25,35,50,70,90])
    plco = plco.loc[plco['smoking_intensity'].notnull()]
    # Remove cigpd_f column
    plco = plco.drop(columns=['cigpd_f'])

    # For duration_smoking: create a column with the duration of smoking. Then remove participant who have no duration of smoking
    plco['duration_smoking'] = plco['cig_years']
    plco = plco.loc[plco['duration_smoking'].notnull()]
    # Remove cig_years column
    plco = plco.drop(columns=['cig_years'])

    # For smoking_quit_time: create a column with the smoking quit time. Then remove participant who have no smoking quit time
    plco['smoking_quit_time'] = plco['cig_stop']
    plco = plco.loc[plco['smoking_quit_time'].notnull()]
    # Remove cig_stop column
    plco = plco.drop(columns=['cig_stop'])

    # For lung cancer, we consider that lung screening is positive if the patient has lung cancer and diagnosis was made in the first 6 years
    plco["lung_cancer"] = 1 * ((plco["lung_cancer"] == 1) & (plco["lung_exitdays"] <= 2190))
    # Remove lung_exitdays column
    plco = plco.drop(columns=['lung_exitdays'])

    # Compute the risk of lung cancer for each participant
    # Create an empty column to store the risk of lung cancer for each participant
    plco["risk"] = 0.
    print("Number of participants in PLCO for PLCOm2012: " + str(len(plco)))

    # Iterate over all the participants
    for index, row in plco.iterrows():
        plco.loc[index, "risk"] = model_plcom2012(row["age"], row["race7"], row["educat"], row["bmi"], row["copd"], row["cancer_hist"],
                                                   row["lung_fh"], row["smoking_status"], row["smoking_intensity"], row["duration_smoking"],
                                                   row["smoking_quit_time"])

    # Compute the performance of the model
    precision, recall, thresholds = precision_recall_curve(plco['lung_cancer'], plco['risk'])
    recall_value_plco = 0.78
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_plco = df.loc[df['recall'] >= recall_value_plco].precision.max() 
    print("On PLCO : For recall = " + str(round(recall_value_plco,5))  + " precision is : " + str(round(max_precision_plco,5)))
    print("-----------------------------------")

    ################################################################
    ###################### COMPARISON ON NLST ######################
    ################################################################
    print("-- Filtering of NLST based on PLCOm2012 --")

    nlst = pd.read_csv(nlst_path, low_memory=False)
    print("Number of participants in NLST for PLCOm2012: " + str(len(nlst)))

    # # Keep participants who never got lung cancer and those who got lung cancer in the first 6 years (2190 days)
    # nlst = nlst.loc[((nlst['candx_days'].isnull()) | (nlst['candx_days'] <= 2190))]

    nlst = nlst[["race", "educat", "height",  "weight", "diagcopd", "num_confirmed", "famfather","fammother", "famchild", "famsister",
                    "fambrother", "cigsmok", "smokeday", "smokeyr", "age_quit","age", "can_scr", "pid", 
                    "cancblad", "cancbrea", "canccerv", "canccolo", "cancesop", "canckidn", "canclary", "canclung", "cancnasa",
                    "cancoral", "cancpanc", "cancphar", "cancstom", "cancthyr", "canctran", "candx_days"]]

    # For age: remove participants which have a non numeric age
    nlst = nlst.loc[nlst['age'].notnull()]
    
    # For race : 4 becomes 1, 3 becomes 4, 6 and above is removed
    # first remove subject which have race=6 or above
    nlst = nlst[nlst["race"]<6]
    nlst["race"] = nlst["race"].replace([4,3],[1,4])
    # remove participant who have no race
    nlst = nlst.loc[nlst['race'].notnull()]

    # For education : 2 becomes 1, 3 becomes 2, 4 becomes 3, 5 becomes 4, 6 becomes 5, 7 becomes 6 and 8 or more is removed
    nlst = nlst[nlst['educat'] < 8]
    nlst["educat"] = nlst["educat"].replace([2,3,4,5,6,7],[1,2,3,4,5,6])
    # Remove participant who have no education
    nlst = nlst.loc[nlst['educat'].notnull()]

    # For bmi : round it to the nearest integer
    nlst.loc[:, 'bmi'] = 703 * nlst['weight'] / (nlst['height']**2) 
    nlst.loc[:, 'bmi'] = nlst['bmi'].round(0)
    # Remove participant who have no bmi
    nlst = nlst.loc[nlst['bmi'].notnull()]
    # Remove weight and height columns
    nlst = nlst.drop(columns=['weight', 'height'])

    # For copd: create a column with binary values : 1 if diagcopd==1 and 0 otherwise
    # Remove participant who have no diagcopd
    nlst = nlst.loc[nlst['diagcopd'].notnull()]
    nlst['copd'] = 1 * (nlst['diagcopd'] == 1)
    # Remove diagcopd column
    nlst = nlst.drop(columns=['diagcopd'])

    # For cancer_hist: create a column with binary values : 1 if sum of the cancer columns is greater than 0 and 0 otherwise 
    nlst['cancer_hist'] = 1 * (nlst[['cancblad', 'cancbrea', 'canccerv', 'canccolo', 'cancesop', 'canckidn', 'canclary', 'canclung', 'cancnasa',
                                    'cancoral', 'cancpanc', 'cancphar', 'cancstom', 'cancthyr', 'canctran']].sum(axis=1) > 0)
    # Remove participant who have no cancer_hist
    nlst = nlst.loc[nlst['cancer_hist'].notnull()]
    # Remove unwanted columns
    nlst = nlst.drop(columns=['cancblad', 'cancbrea', 'canccerv', 'canccolo', 'cancesop', 'canckidn', 'canclary', 'cancnasa',
                                    'cancoral', 'cancpanc', 'cancphar', 'cancstom', 'cancthyr', 'canctran'])

    # For family_hist_lung_cancer: binary value : 1 if at least one of the famfather, fammother, famchild, famsister, fambrother is 1
    nlst["lung_fh"] = nlst[["famfather","fammother", "famchild", "famsister", "fambrother"]].max(axis=1)
    # Remove participant who have no family_hist_lung_cancer
    nlst = nlst.loc[nlst['lung_fh'].notnull()]
    # Drop the columns famfather, fammother, famchild, famsister, fambrother
    nlst = nlst.drop(columns=["famfather","fammother", "famchild", "famsister", "fambrother"])

    # For smoking_status: create a column with binary values : 1 if cigsmok==1 and 0 otherwise
    # Remove participant who have no cigsmok
    nlst = nlst.loc[nlst['cigsmok'].notnull()]
    nlst['smoking_status'] = 1 * (nlst['cigsmok'] == 1)
    # Remove cigsmok column
    nlst = nlst.drop(columns=['cigsmok'])

    # For smoking_intensity: create a column with the number of cigarettes smoked per day. Then remove participant who have no smoking intensity
    nlst['smoking_intensity'] = nlst['smokeday']
    nlst = nlst.loc[nlst['smoking_intensity'].notnull()]
    # Remove smokeday column
    nlst = nlst.drop(columns=['smokeday'])

    # For duration_smoking: create a column with the duration of smoking. Then remove participant who have no duration of smoking
    nlst['duration_smoking'] = nlst['smokeyr']
    nlst = nlst.loc[nlst['duration_smoking'].notnull()]
    # Remove smokeyr column
    nlst = nlst.drop(columns=['smokeyr'])

    # For smoking_quit_time: create a column with the years since the person has quit smoking = age - age_quit or 0 if age_quit is Null
    nlst['smoking_quit_time'] = nlst['age'] - nlst['age_quit']
    nlst.loc[nlst['age_quit'].isnull(), 'smoking_quit_time'] = 0
    # Replace negative values by 0
    nlst.loc[nlst['smoking_quit_time'] < 0, 'smoking_quit_time'] = 0
    # Remove column age_quit
    nlst = nlst.drop(columns=['age_quit'])

    # For cancer screening : 1 if can_scr>0 and 0 otherwise
    nlst["lung_cancer"] = nlst['candx_days'].apply(pd.to_numeric)
    nlst["lung_cancer"] = nlst["lung_cancer"].apply(lambda x: 1 if pd.notnull(x) else 0)
    nlst["lung_cancer"] = nlst["lung_cancer"] * (nlst['candx_days']<=2190)
    # Drop the column candx_days
    nlst = nlst.drop(columns=["candx_days"])

    # Compute the risk of lung cancer for each participant
    # Create an empty column to store the risk of lung cancer for each participant
    nlst["risk"] = 0.
    print("Number of participants in NLST for PLCOm2012: " + str(len(nlst)))

    # Iterate over all the participants
    for index, row in nlst.iterrows():
        nlst.loc[index, "risk"] = model_plcom2012(row["age"],row["race"], row["educat"], row["bmi"], row["copd"], row["cancer_hist"],
                                                   row["lung_fh"], row["smoking_status"], row["smoking_intensity"], row["duration_smoking"],
                                                   row["smoking_quit_time"])
        
    # Compute the performance of the model
    precision, recall, thresholds = precision_recall_curve(nlst['lung_cancer'], nlst['risk'])
    recall_value_nlst = 0.9886055344546935
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_nlst = df.loc[df['recall'] >= recall_value_nlst].precision.max() 
    print("On NLST : For recall = " + str(round(recall_value_nlst,5))  + " precision is : " + str(round(max_precision_nlst,5)))
    print("-----------------------------------")

    #######################################################################
    ###################### COMPARISON WITH XGB MODEL ######################
    #######################################################################
    print("-- Comparison with XGB model --")
    print("In this scenario we look at the intersection of filtering done by PLCOm2012 and the filtering done in our project.")
    print("The comparison of performance is done on that subset of data.")

    # Load the preprocessed data
    nlst_xgb = pd.read_csv(args.nlst_preprocessed)
    plco_xgb = pd.read_csv(args.plco_preprocessed)

    # Ad the index column in plco and nlst
    plco["index"] = plco.index
    nlst["index"] = nlst.index
    
    # We keep only the data in preprocessed plco for which the index is in plco (same for nlst)
    plco_xgb = plco_xgb.loc[plco_xgb['index'].isin(plco['index'])]
    nlst_xgb = nlst_xgb.loc[nlst_xgb['index'].isin(nlst['index'])]
    print("Number of participants in PLCO for PLCOm2012 and XGB: " + str(len(plco_xgb)))
    print("Number of participants in NLST for PLCOm2012 and XGB: " + str(len(nlst_xgb)))
    # Same thing is done on the plco dataset using the plco_xgb dataset
    plco = plco.loc[plco['index'].isin(plco_xgb['index'])]
    nlst = nlst.loc[nlst['index'].isin(nlst_xgb['index'])]

    columns_to_keep = ['cig_years', 'age', 'pack_years', 'ssmokea_f', 'cig_stat', 'cigpd_f', 'bmi', 'lung_fh', 'lung_cancer']
    plco_xgb = plco_xgb[columns_to_keep]
    nlst_xgb = nlst_xgb[columns_to_keep]

    # Evaluation of the XGB model on NLST
    ## Load the XGB model
    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)
    # Compute the risk of lung cancer for each participant using XGB
    plco_xgb["risk"] = xgb_model.predict_proba(plco_xgb.drop(columns=['lung_cancer']))[:,1]
    # Compute the performance of the XGB model
    precision, recall, thresholds = precision_recall_curve(plco_xgb['lung_cancer'], plco_xgb['risk'])
    recall_value_plco = 0.78
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_plco = df.loc[df['recall'] >= recall_value_plco].precision.max() 
    print("On PLCO with XGB : For recall = " + str(round(recall_value_plco,5))  + " precision is : " + str(round(max_precision_plco,5)))

    # Evaluation of the PLCOm2012 model on PLCO
    precision, recall, thresholds = precision_recall_curve(plco['lung_cancer'], plco['risk'])
    recall_value_plco = 0.78
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_plco = df.loc[df['recall'] >= recall_value_plco].precision.max() 
    print("On PLCO with PLCOm2012: For recall = " + str(round(recall_value_plco,5))  + " precision is : " + str(round(max_precision_plco,5)))

    ## Evaluation of the XGB model on NLST
    # Compute the risk of lung cancer for each participant
    nlst_xgb["risk"] = xgb_model.predict_proba(nlst_xgb.drop(columns=['lung_cancer']))[:,1]
    # Compute the performance of the model
    precision, recall, thresholds = precision_recall_curve(nlst_xgb['lung_cancer'], nlst_xgb['risk'])
    recall_value_nlst = 0.9886055344546935
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_nlst = df.loc[df['recall'] >= recall_value_nlst].precision.max()
    print("On NLST with XGB : For recall = " + str(round(recall_value_nlst,5))  + " precision is : " + str(round(max_precision_nlst,5)))

    # Evaluation of the PLCOm2012 model on NLST
    precision, recall, thresholds = precision_recall_curve(nlst['lung_cancer'], nlst['risk'])
    recall_value_nlst = 0.9886055344546935
    df = pd.concat([pd.DataFrame(precision, columns=['precision']), 
            pd.DataFrame(recall,columns=['recall']), 
            pd.DataFrame(thresholds,columns=['thresholds'])], axis=1)
    max_precision_nlst = df.loc[df['recall'] >= recall_value_nlst].precision.max()
    print("On NLST with PLCOm2012: For recall = " + str(round(recall_value_nlst,5))  + " precision is : " + str(round(max_precision_nlst,5)))
    print("-----------------------------------")
    
    return None


if __name__ == "__main__":
    main()