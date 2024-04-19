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

    # We first remove all non-smokers or non-previous smokers
    plco = plco.loc[plco.cig_stat > 0]

    # We remove patients who died of something else than lung cancer
    plco = plco.loc[plco['d_dthl']!=0]
    nlst = nlst.loc[nlst['finaldeathlc']!=0]

    # We remove patients who were study for less than 5 years (1827 jours) 
    plco = plco[((plco['lung_exitstat']!=1) & (plco['lung_exitdays']>1827)) | (plco['lung_exitstat']==1)]
    nlst = nlst[((nlst['scr_group']!=1) & (nlst['fup_days']>1827)) | (nlst['scr_group']==1)]

    # Uniformisation of both datasets
    plco = plco[["age", "sex", "height_f", "weight_f", "race7", "ssmokea_f", "cig_stat", "cigar", "pipe", "pack_years", "smokea_f", "cigpd_f","cig_years", "bronchit_f",
                    "diabetes_f", "emphys_f", "hearta_f", "hyperten_f", "stroke_f", "lung_fh","lung_cancer"
                ]]
    plco["race7"] = plco["race7"].replace(3,1)
    plco["lung_fh"] = plco["lung_fh"].replace(9,0)
    nlst2 = nlst[["age", "gender", "height",  "weight", "race", "age_quit", "cigsmok", "cigar", "pipe", "pkyr", "smokeage", "smokeday", "smokeyr", "agechro", "diagdiab",
            "diagemph", "diaghear", "diaghype", "diagstro",
            ]]
    nlst2["lung_fh"] = nlst[["famfather","fammother", "famchild", "famsister", "fambrother"]].max(axis=1)
    nlst2["can_scr"] = 1 * (nlst["can_scr"] > 0)
    nlst2["race"] = nlst2["race"].replace([3,4,6,95,96,98,99],[4,6,7,7,7,7,7])
    nlst2['cigsmok'] = nlst2["cigsmok"].replace(0,2)
    nlst=nlst2

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

    # Save the data
    plco.to_csv(os.path.join(output_path, "preprocessed_plco.csv"), index=False)
    nlst.to_csv(os.path.join(output_path, "preprocessed_nlst.csv"), index=False)
    
    return None


if __name__ == "__main__":
    main()