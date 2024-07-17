import pandas as pd
import os
import numpy as np
from src.utils import utils
from src import loading
import logging
from datetime import datetime


def cleanup_patients(
    df: pd.DataFrame,
    col_date_of_death,
    col_date_of_birth,
    col_deceased_indicator: str,
    deceased_value: str or int,
) -> pd.DataFrame:
    """Removes patient recorded as dead after turning 89 as their dates of birth have been changed.

    Args:
        df (pd.DataFrame): Pandas dataframe
        col_deceased_indicator (str): Name of the oclumn containing the flag alive/deceased
        deceased_value (str or int): Value of the flag when patient is deceased

    Returns:
        pd.DataFrame: Cleaned pandas dataframe
    """
    # Create column with age at death
    df["AGE_AT_DEATH"] = np.where(
        df[col_deceased_indicator] == deceased_value,
        utils.compute_timegap(df, col_date_of_death, col_date_of_birth),
        np.nan,
    )

    # Clean up
    df = df[~(df["AGE_AT_DEATH"] < 0)]
    return df


def prepare_patients(
    df_patients: pd.DataFrame,
    col_date_of_death: str,
    col_date_of_birth: str,
    col_deceased_indicator: str,
) -> pd.DataFrame:
    """Prepares the patients table.
    Namely removing patients patients recorded as dead after turning 89 as their DOB have been changed

    Args:
        df_patients (pd.DataFrame): patients table
        col_date_of_death (str): Name of column of DOD
        col_date_of_birth (str): Name of column of DOB
        col_deceased_indicator (str): Name of column of deceased indicator

    Returns:
        pd.DataFrame: Patients table prepared
    """
    # Preprocess patients
    df_patients = utils.turn_to_datetime(
        df_patients, [col_date_of_death, col_date_of_birth]
    )
    df_patients = cleanup_patients(
        df_patients, col_date_of_death, col_date_of_birth, col_deceased_indicator, 1
    )
    df_patients = df_patients.replace(
        {"F": 0, "M": 1}
    )
    logging.info("Patients table preprocessed")

    return df_patients
