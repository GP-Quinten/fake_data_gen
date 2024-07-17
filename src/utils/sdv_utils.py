import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata.single_table import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport


def build_dict_metadata_from_df(
    df: pd.DataFrame, primary_key: str, discrete_columns: list
) -> dict:
    """Creates a dictionnary to use as metadata for SDV report.

    Args:
        df (pd.DataFrame): Dataframe, can be real or synthetic.
        primary_key (str): Primary key to initiate metadata dict.
        discrete_columns (list): List of discrete columns.

    Returns:
        dict: metadata dictionnary
    """
    start_dict = {"primary_key": primary_key, "columns": {}}

    for col in discrete_columns:
        start_dict["columns"][col] = {"sdtype": "categorical"}

    numerical_columns = [col for col in df.columns if col not in discrete_columns]
    for col in numerical_columns:
        start_dict["columns"][col] = {"sdtype": "numerical"}
    return start_dict


def get_metadata_from_dict(dict_metadata):
    metadata = SingleTableMetadata()
    for key in dict_metadata.keys():
        key = str(key)
        setattr(metadata, key, dict_metadata[key])
    return metadata


def get_sdv_report(
    df_real: pd.DataFrame, df_synth: pd.DataFrame, dict_metadata: dict
) -> QualityReport:
    """Retrieves a sdv Quality report

    Args:
        df_real (pd.DataFrame): preprocessed original cohort data, 1 row/ patient & 1 col/ variable
        df_synth (pd.DataFrame): simulated cohort data, 1 row/ patient & 1 col/ variable
        dict_metadata (dict): metadata dictionnary

    Returns:
        QualityReport: sdv quality report
    """

    # create sdv metadata dictionary
    sdv_metadata = get_metadata_from_dict(dict_metadata)

    # sdv quality report
    sdv_report = evaluate_quality(df_real, df_synth, sdv_metadata)

    return sdv_report
