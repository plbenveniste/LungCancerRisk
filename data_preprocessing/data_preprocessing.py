"""
In this file we perform data preprocessing on the data. 

Args:
    --nlst-path: Path to the NLST file
    --plco-path: Path to the PLCO file
    --output-path: Path to save the preprocessed data

Returns:
    None

Example:
    python data_preprocessing.py --nlst-path data/NLST.csv --plco-path data/PLCO.csv --output-path data/preprocessed_data.csv

TODO:
    - add a logger where we can save the evolution of the size of the dataset

Pierre-Louis Benveniste
"""
import argparse
import pandas as pd
import os


def get_parser():
    """
    This function parses the arguments passed to the script.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('--nlst-path', type=str, help='Path to the NLST file')
    parser.add_argument('--plco-path', type=str, help='Path to the PLCO file')
    parser.add_argument('--output-path', type=str, help='Path to save the preprocessed data')
    return parser


def main():
    """
    This function performs data preprocessing on the data.

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
    output_path = args.output_path

    # Loading of both datasets
    plco = pd.read_csv(plco_path)
    nlst = pd.read_csv(nlst_path, low_memory=False)
    print("Number of patients in PLCO: ", plco.shape[0])
    print("Number of patients in NLST: ", nlst.shape[0])

    # We first remove all non-smokers or non-previous smokers
    plco = plco.loc[plco.cig_stat > 0]
    print("Number of patients in PLCO after removing non-smokers: ", plco.shape[0])

    # We remove patients who died of something else than lung cancer
    plco = plco.loc[plco['d_dthl']!=0]
    nlst = nlst.loc[nlst['finaldeathlc']!=0]
    print("Number of patients in PLCO after removing patients who died of something else than lung cancer: ", plco.shape[0])
    print("Number of patients in NLST after removing patients who died of something else than lung cancer: ", nlst.shape[0])

    # Censored data removal:
    ## Either subjects didn't have cancer and were study for at least than 6 years
    ## Or subjects had cancer
    plco = plco[((plco['lung_cancer']!=1) & (plco['lung_exitdays']>2190)) | (plco['lung_cancer']==1)]
    nlst = nlst[((pd.to_numeric(nlst['candx_days'], errors='coerce').isnull()) & (nlst['fup_days']>2190)) | (pd.to_numeric(nlst['candx_days'], errors='coerce').notnull())]
    print("Number of patients in PLCO after removing patients who were study for less than 6 years: ", plco.shape[0])
    print("Number of patients in NLST after removing patients who were study for less than 6 years: ", nlst.shape[0])

    # Uniformisation of both datasets
    ## Uniformisation of PLCO
    plco = plco[["age", "sex", "height_f", "weight_f", "race7", "ssmokea_f", "cig_stat", "cigar", "pipe", "pack_years", "smokea_f", "cigpd_f","cig_years", "bronchit_f",
                    "diabetes_f", "emphys_f", "hearta_f", "hyperten_f", "stroke_f", "lung_fh", "lung_cancer", "candxdaysl"
                ]]
    plco["race7"] = plco["race7"].replace(3,1)
    plco["lung_fh"] = plco["lung_fh"].replace(9,0)

    # We consider that lung screening is positive if the patient has lung cancer and diagnosis was made in the first 6 years
    plco["lung_cancer"] = plco["lung_cancer"].apply(lambda x: 1 if x==1 else 0)
    print("Number of patients in PLCO with lung cancer: ", plco[plco["lung_cancer"]==1].shape[0])
    plco["lung_cancer"] = plco["lung_cancer"] * (plco["candxdaysl"]<=2190)
    print("Number of patients in PLCO with lung cancer and diagnosis in the first 6 years: ", plco[plco["lung_cancer"]==1].shape[0])
    # We remove the candxdaysl column
    plco = plco.drop(columns=["candxdaysl"])

    ## Uniformisation of NLST
    nlst_temp = nlst[["age", "gender", "height",  "weight", "race", "age_quit", "cigsmok", "cigar", "pipe", "pkyr", "smokeage", "smokeday", "smokeyr", "agechro", "diagdiab",
            "diagemph", "diaghear", "diaghype", "diagstro", 'candx_days'
            ]]
    ### Creation of the lung_cancer variable
    nlst_temp["lung_cancer"] = nlst_temp['candx_days'].apply(pd.to_numeric)
    nlst_temp["lung_cancer"] = nlst_temp["lung_cancer"].apply(lambda x: 1 if pd.notnull(x) else 0)
    print("Number of patients in NLST with lung cancer: ", nlst_temp[nlst_temp["lung_cancer"]==1].shape[0])
    nlst_temp["lung_cancer"] = nlst_temp["lung_cancer"] * (nlst_temp['candx_days']<=2190)
    print("Number of patients in NLST with lung cancer and diagnosis in the first 6 years: ", nlst_temp[nlst_temp["lung_cancer"]==1].shape[0])
    # We remove the candx_days column
    nlst_temp = nlst_temp.drop(columns=["candx_days"])

    nlst_temp["lung_fh"] = nlst[["famfather","fammother", "famchild", "famsister", "fambrother"]].max(axis=1)
    nlst_temp["race"] = nlst_temp["race"].replace([3,4,6,95,96,98,99],[4,6,7,7,7,7,7])
    nlst_temp['cigsmok'] = nlst_temp["cigsmok"].replace(0,2)
    nlst=nlst_temp

    # renaming columns to same name
    change_columns = {
            "age": "age",
            "gender": "sex", 
            "height": "height_f",
            "weight": "weight_f",
            "race": "race7",
            "age_quit": "ssmokea_f",
            "cigsmok": "cig_stat",
            "cigar": "cigar",
            "pipe": "pipe",
            "pkyr": "pack_years",
            "smokeage": "smokea_f",
            "smokeday": "cigpd_f",
            "smokeyr": "cig_years",
            "agechro": "bronchit_f",
            "diagdiab": "diabetes_f",
            "diagemph": "emphys_f",
            "diaghear": "hearta_f",
            "diaghype": "hyperten_f",
            "diagstro": "stroke_f",
            "can_scr": "lung_cancer",
            "lung_fh": "lung_fh"
    }
    nlst = nlst.rename(columns=change_columns)

    # Add BMI column
    plco.loc[:, 'bmi'] = plco['weight_f'] / plco['height_f']**2 * 703
    nlst['bmi'] = nlst['weight_f'] / nlst['height_f']**2 * 703
    print("Final number of columns in PLCO: ", plco.shape[1])
    print("Final number of columns in NLST: ", nlst.shape[1])

    # Save the data
    plco.to_csv(os.path.join(output_path, "preprocessed_plco.csv"), index=False)
    nlst.to_csv(os.path.join(output_path, "preprocessed_nlst.csv"), index=False)
    print("Data saved in ", output_path)
    
    return None


if __name__ == "__main__":
    main()