import pandas as pd
import os
import numpy as np
from src.utils import utils
import logging


def compute_followup(
    df: pd.DataFrame,
    col_date_of_death: str,
    col_flag_value: int,
    new_column_name: str,
    alive_flag_value: int,
) -> pd.DataFrame:
    """Computes followup period for each patient.
    For dead patients, the followup period is equivalent to the timegap between their first admission date and their date of death. For alive patients, it is the timegap between their last and first admission.

    Args:
        df (pd.DataFrame): Pandas dataframe
        col_date_of_death (str): Name of the column containing the date of death
        col_flag_value (int): Name of the column containing the flags defining if a patient is alive or not
        new_column_name (str): Name of the new column containing the followup period
        alive_flag_value (int): Value of the flag that indicates a patient is alive

    Returns:
        pd.DataFrame: Pandas dataframe with new column containing the followup information
    """
    # Computes follow up period
    df[new_column_name] = np.where(
        df[col_flag_value] == alive_flag_value,
        ((df["LAST_ADMISSION_DATE"] - df["FIRST_ADMISSION_DATE"]).dt.days),
        ((df[col_date_of_death] - df["FIRST_ADMISSION_DATE"]).dt.days),
    )

    # Cleaning followup : if followup <0 -> change it to 0
    df[new_column_name] = np.where(
        df[new_column_name] < 0,
        0,
        df[new_column_name],
    )

    return df


def create_final_demog_df(df: pd.DataFrame, col_to_keep: list) -> pd.DataFrame:
    """Creates the final dataframe of demographics

    Args:
        df (pd.DataFrame): Pandas dataframe
        col_to_keep (list): Columns we wish to keep from the dataframe

    Returns:
        pd.DataFrame: Final dataframe.
    """
    return df[col_to_keep].drop_duplicates()


def prepare_demog(
    df_patients: pd.DataFrame,
    df_adm: pd.DataFrame,
    patient_id: str,
    col_to_keep_patients: list,
    col_to_keep_adm: list,
    col_date_of_birth: str,
    col_date_of_death: str,
    col_deceased_indicator: str,
    col_to_keep_demog: str,
) -> pd.DataFrame:
    """
    Prepares the dataframe of demograhics.
    Merging patients with admission, computing AGE_AT_LAST_ADMISSION, computing FOLLOWUP_PERIOD and cleaning.

    Args:
        df_patients (pd.DataFrame): Pandas dataframe of patients table prepared
        df_adm (pd.DataFrame): Pandas dataframe of admission table prepared
        patient_id (str): Name of column of patient id
        col_to_keep_patients (list): Columns to keep in patients table
        col_to_keep_adm (list): Columns to keep in admission table
        col_date_of_birth (str): Name of column of DOB
        col_date_of_death (str): Name of column of DOD
        col_deceased_indicator (str): Name of column with deceased indicator
        col_to_keep_demog (str): Columns to keep in demographics table

    Returns:
        Pandas Dataframe: Prepared demographics table
    """
    # Merge on IDs
    df_demog = utils.merge_dataframes(
        df_patients,
        df_adm,
        patient_id,
        "inner",
        col_to_keep_patients,
        col_to_keep_adm,
    )

    ## Compute age at last admission
    df_demog["AGE_AT_LAST_ADMISSION"] = utils.compute_timegap(
        df_demog, "LAST_ADMISSION_DATE", col_date_of_birth
    )

    ## Clean up patients with a shifted birthdate
    df_demog = utils.cleanup_col(df_demog, "AGE_AT_LAST_ADMISSION", 0)

    ## Compute followup period
    df_demog = compute_followup(
        df_demog,
        col_date_of_death,
        col_deceased_indicator,
        "FOLLOWUP_PERIOD",
        0,
    )

    ## Retrieve columns of interest and drop duplicates
    df_demog_final = create_final_demog_df(df_demog, col_to_keep_demog)

    logging.info("Demogs ready")
    return df_demog_final
