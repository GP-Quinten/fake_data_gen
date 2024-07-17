import pandas as pd
import logging


def most_prevalent_diags(
    df_diag: pd.DataFrame, min_prevalence: int, patient_id: str
) -> list:
    """Retrieves the diags that have a prevalence > min_prevalence from the table of diags df_diags

    Args:
        df_diag (pd.DataFrame): diags table
        min_prevalence (int): minimum of prevalence
        patient_id (str): Name of column containing patient id

    Returns:
        list: list of diags to filter on
    """
    if patient_id in df_diag.columns:
        df_diag = df_diag.drop(patient_id, axis=1)
    df_inverted = df_diag.T

    df_inverted["PREVALENCE"] = df_inverted.sum(axis=1) / len(df_diag) * 100
    df_prevalence_min = df_inverted[df_inverted["PREVALENCE"] >= min_prevalence]
    col_to_select = df_prevalence_min.index.tolist()
    return col_to_select


def get_most_prevalent_diags(
    df_diag: pd.DataFrame, min_prevalence: int, patient_id: str
) -> pd.DataFrame:
    """Filters the diags table on diags that have a prevalence of minimum min_prevalence

    Args:
        df_diag (pd.DataFrame): diags table
        min_prevalence (int): minimum of prevalence
        patient_id (str): Name of column containing patient id

    Returns:
        pd.DataFrame: Returns a table with only diags that have a min_prev > min_prevalence
    """
    col_to_keep = most_prevalent_diags(df_diag, min_prevalence, patient_id) + [
        patient_id
    ]
    return df_diag.loc[:, df_diag.columns.isin(col_to_keep)]


def prepare_diag(
    df_diag: pd.DataFrame,
    patient_id: str,
    col_diag: str,
    min_prev,
    preproc_type="binary",
) -> pd.DataFrame:
    """
    Computation depends on preproc - type:
    - if preproc_type='binary' : Computes the binary matrix of the table diags : 1 row by patient, each column is a diag type with value 0 is the patient has never been diagnosed with this diag, 1 if yes.
    - if preproc_type='count' : Computes the binary matrix of the table diags : 1 row by patient, each column is a diag type.
    df_diags[i,j] is the number of times patient i has been diagnosed with diagnosis j.

    Args:
        df_diag (pd.DataFrame): Pandas dataframe
        col_PTID (str): Name of the column containing the PTIDs of the patients
        col_diag (str): Name of the column containing the diags
        preproc_type (str): Type of preprocessing we want to do

    Returns:
        df_diag (pd.DataFrame): Binary (or count) matrix
    """

    # Pre-preprocessing
    df_diag = (
        df_diag[[patient_id, col_diag]]
        .dropna(subset=col_diag)
        .rename(columns={col_diag: "ICD"})
    )  # renaming for columns names when using get_dummies
    df_diag["ICD"] = (
        df_diag["ICD"].astype(str).map(lambda x: x[:3])
    )  # truncate to first 3 characters

    if preproc_type == "binary":
        # Compute the binary matrix
        df_diag = (
            pd.get_dummies(df_diag, columns=["ICD"])
            .groupby([patient_id], as_index=False)
            .agg("max")
        )

    else:
        # Compute the binary matrix
        df_diag = (
            pd.get_dummies(df_diag, columns=["ICD"])
            .groupby([patient_id], as_index=False)
            .agg("sum")
        )

    logging.info(f"Diags prepared with type '{preproc_type}'")
    df_diag = get_most_prevalent_diags(df_diag, min_prev, patient_id)
    logging.info(f"Diags filtered on min prev {min_prev}")
    return df_diag
