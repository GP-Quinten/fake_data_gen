import pandas as pd
from itertools import chain


def filter_on_codelist(df: pd.DataFrame, map: dict, col_to_filter: str) -> pd.DataFrame:
    """Filters table on a codelist

    Args:
        df (pd.DataFrame): Pandas dataframe to filter
        map (dict): Dict with values to filter on
        col_to_filter (str): Column to filter on

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    value_list = list(set(chain.from_iterable(map.values())))
    return df[df[col_to_filter].isin(value_list)]


def compute_labs_mean(
    df: pd.DataFrame, col_num_values: str, col_labels: str, col_patient_id: str
) -> pd.DataFrame:
    """Computes mean of lab values

    Args:
        df (pd.DataFrame): Pandas dataframe with labs values for each patients
        col_num_values (str): Column with numerical value of labs
        col_labels (str): Column with lab names
        col_patient_id (str): Column with patient id

    Returns:
        pd.DataFrame: Dataframe with one row / PTID, one column per lab
    """
    # Filtering out values with a null numerical value
    # Lab values can be numerical or indicated as ranges (above, normal, below) so this allows to keep only numerical values

    df = df[~df[col_num_values].isnull()]
    mean_df = (
        df.groupby([col_patient_id, col_labels])
        .agg({col_num_values: "mean"})
        .reset_index()
    )
    mean_df_pivot = mean_df.pivot(
        index=col_patient_id, columns=col_labels, values=col_num_values
    ).reset_index()
    return mean_df_pivot


def prepare_labs(
    df_labs: pd.DataFrame,
    df_labs_desc: pd.DataFrame,
    loinc_map: dict,
    col_loinc_codes: str,
    col_lab_id: str,
    col_label: str,
    col_num_values: str,
    patient_id: str,
) -> pd.DataFrame:
    """Combines previous functions to prepare lab table

    Args:
        df_labs (pd.DataFrame): Pandas dataframe of lab values
        df_labs_desc (pd.DataFrame): Pandas dataframe of lab descriptions
        loinc_map (dict): Map of LOINC codes to lab names
        col_loinc_codes (str): Column with LOINC codes
        col_lab_id (str): Column with lab codes
        col_label (str): Column with labels
        col_num_values (str): Column with numerical value of labs
        patient_id (str): Column with patient_id

    Returns:
        pd.DataFrame: _description_
    """
    labs_desc_filtered = filter_on_codelist(df_labs_desc, loinc_map, col_loinc_codes)
    labels_to_code = {value: key for key in loinc_map for value in loinc_map[key]}
    labs_desc_filtered[col_label] = labs_desc_filtered[col_loinc_codes].map(
        labels_to_code
    )

    labs_of_interest = df_labs.merge(
        labs_desc_filtered[[col_lab_id, col_label, col_loinc_codes]],
        on=col_lab_id,
        how="inner",
    )

    mean_labs = compute_labs_mean(
        labs_of_interest, col_num_values, col_label, patient_id
    )
    return mean_labs
