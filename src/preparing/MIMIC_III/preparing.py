import pandas as pd
import numpy as np
import logging
from src.preparing.MIMIC_III.preparing_patients import prepare_patients
from src.preparing.MIMIC_III.preparing_adm import prepare_adm
from src.preparing.MIMIC_III.preparing_demog import prepare_demog
from src.preparing.MIMIC_III.preparing_diag import prepare_diag
from src.preparing.MIMIC_III.preparing_labs import prepare_labs


def remove_columns_with_high_nan(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Removes columns from a DataFrame with more than X% NaN values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): The threshold percentage (between 0 and 100) for NaN values.

    Returns:
    - pd.DataFrame: A DataFrame with columns containing more than X% NaN values removed.
    """
    # Calculate the maximum number of NaN values allowed for each column
    max_nan_values = threshold / 100 * len(df)

    # Filter columns based on NaN values count
    filtered_columns = df.columns[df.isna().sum() <= max_nan_values]

    # Create a new DataFrame with the selected columns
    cleaned_df = df[filtered_columns]

    return cleaned_df


def preparing(
    df_patients: pd.DataFrame,
    df_adm: pd.DataFrame,
    df_diag: pd.DataFrame,
    df_labs: pd.DataFrame,
    df_labs_desc: pd.DataFrame,
    patient_id: str,
    col_date_of_death: str,
    col_date_of_birth: str,
    col_deceased_indicator: str,
    col_admission_time: str,
    col_ethnicity: str,
    col_to_keep_patients: list,
    col_to_keep_adm: list,
    col_to_keep_demog: list,
    col_diag: str,
    min_prev: int,
    loinc_map: dict,
    col_loinc_codes: str,
    col_lab_id: str,
    col_label: str,
    col_num_values: str,
    remove_nan: bool,
) -> pd.DataFrame:
    """
    Prepares the input of our model : takes 3 tables of MIMIC_III database (patients, admission and diagnosis) and makes some preparing and merging of those 3 tables.
    6 steps/functions :
    - prepare_patients to prepare the patients table
    - prepare_admissions to prepare the admission table
    - prepare_demog to prepare the demographics table (combining/merging patients and admission)
    - prepare_diags to prepare the diagnosis table
    - prepare_labs to prepare the labs table
    - merging all

    Args:
        df_patients (pd.DataFrame): Pandas dataframe of table patients
        df_adm (pd.DataFrame): Pandas dataframe of table admission
        df_diag (pd.DataFrame): Pandas dataframe of table diag
        df_labs (pd.DataFrame): Pandas dataframe of labs table
        patient_id (str): Name of the column containing the patient id in all 3 tables
        col_date_of_death (str): Name of the column containing DOD in patients table
        col_date_of_birth (str): Name of the column containing DOB in patients table
        col_deceased_indicator (str): Name of the column containing alive or dead patient in patients table
        col_admission_time (str): Name of the column containing the time of admission in admission table
        col_ethnicity (str): Name of the column containing the ethnicity in admission table
        col_to_keep_patients (list): Columns to keep in patients table
        col_to_keep_adm (list): Columns to keep in admission table
        col_to_keep_demog (list): Columns to keep in demographics table
        col_diag (str): Name of the column containing the diags in diags table
        min_prev (int): Minimum of percentage of prevalence to filter on diags table
        loinc_map (dict): Map of LOINC codes to lab names
        col_loinc_codes (str): Column with LOINC codes
        col_lab_id (str): Column with lab codes
        col_label (str): Column with labels
        col_num_values (str): Column with numerical value of labs
        remove_nan (bool): Wether or not to remove missing values from dataset

    Returns:
        pd.DataFrame: Pandas dataframe prepared : as input for the modelling.
    """

    df_patients = prepare_patients(
        df_patients, col_date_of_death, col_date_of_birth, col_deceased_indicator
    )

    df_adm = prepare_adm(df_adm, col_admission_time, col_ethnicity, patient_id)

    df_demogs = prepare_demog(
        df_patients,
        df_adm,
        patient_id,
        col_to_keep_patients,
        col_to_keep_adm,
        col_date_of_birth,
        col_date_of_death,
        col_deceased_indicator,
        col_to_keep_demog,
    )

    df_diag = prepare_diag(
        df_diag, patient_id, col_diag, min_prev, preproc_type="binary"
    )

    df_labs = prepare_labs(
        df_labs,
        df_labs_desc,
        loinc_map,
        col_loinc_codes,
        col_lab_id,
        col_label,
        col_num_values,
        patient_id,
    )

    # Mergin demogs with diags
    df = df_demogs.merge(df_diag, on=patient_id, how="inner")
    df_all = df.merge(df_labs, on=patient_id, how="left")
    if remove_nan:
        df_all = remove_columns_with_high_nan(df_all, 50)
        df_all = df_all.fillna(df_all.median())

    logging.info("Real data computed")

    return df_all
