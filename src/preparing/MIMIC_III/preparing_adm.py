import logging
import pandas as pd
import numpy as np
from src.utils import utils

def most_frequent_feature(
    df: pd.DataFrame,
    col_name_feature: str,
    patient_id: str,
    replace: bool = True,
    name_of_new_column: str = None,
) -> pd.DataFrame:
    """Encodes feature as 1 if most frequent, 0 if different.

    Args:
        df (pd.DataFrame): Pandas dataframe
        col_name_feature (str): Name of the column of interest
        replace (bool, optional): Is the column replaced or not (False creates a new one). Defaults to True.
        name_of_new_column (str, optional): If replace==False, creates a new column named name_of_new_column. Defaults to None.

    Returns:
        pd.DataFrame: Pandas dataframe with feature encoded.
    """
    most_frequent = df[col_name_feature].mode()[0]
    if not replace and name_of_new_column is not None:
        df[name_of_new_column] = np.where(df[col_name_feature] == most_frequent, 1, 0)

    elif not replace and name_of_new_column is None:
        print("Enter a new column name")

    else:
        df[col_name_feature] = np.where(df[col_name_feature] == most_frequent, 1, 0)

    df = df.sort_values(col_name_feature, ascending=False)
    df = df.drop_duplicates(subset=[patient_id, col_name_feature], keep="first")
    df[col_name_feature] = df[col_name_feature].astype("category")

    return df


def compute_first_and_last_admittime(
    df: pd.DataFrame,
    col_admission_date: str,
    patient_id: str,
) -> pd.DataFrame:
    """Creates min and max admission dates column

    Args:
        df (pd.DataFrame): Pandas dataframe
        col_admission_date (str): Name of the column containing the admission dates

    Returns:
        pd.DataFrame: Pandas dataframe with two new columns
    """
    df["FIRST_ADMISSION_DATE"] = df.groupby(patient_id)[col_admission_date].transform(
        "min"
    )
    df["LAST_ADMISSION_DATE"] = df.groupby(patient_id)[col_admission_date].transform(
        "max"
    )
    return df


def prepare_adm(
    df_adm: pd.DataFrame,
    col_admission_time: pd.DataFrame,
    col_ethnicity: str,
    patient_id: str,
):
    ## Preprocess admissions
    df_adm = utils.turn_to_datetime(df_adm, [col_admission_time])
    # maps patients whose ethnicity is the most common as 1, and any other as 0
    df_adm = most_frequent_feature(df_adm, col_ethnicity, patient_id)
    df_adm = compute_first_and_last_admittime(df_adm, col_admission_time, patient_id)
    logging.info("Admissions table preprocessed")
    return df_adm
