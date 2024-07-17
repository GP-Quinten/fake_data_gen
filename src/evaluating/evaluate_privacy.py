import pandas as pd
from sdmetrics.single_table import CategoricalCAP


def privacy_check(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    key_fields: list,
    sensitive_fields: list,
) -> pd.DataFrame:
    """Computes privacy scores considering an attacker wants to guess one of the sensitives fields columns and knows the key fields columns.

    Args:
        df_real (pd.DataFrame): Dataframe with real data.
        df_synth (pd.DataFrame): Dataframe with synthetized data.
        key_fields (list): Columns known by the attacker.
        sensitive_fields (list): Column the attacker tries to guess.

    Returns:
        pd.DataFrame: Dataframe with privacy scores for each sensitive column.
    """
    df_scores = pd.DataFrame()
    for sensitive_col in sensitive_fields:
        score = CategoricalCAP.compute(
            real_data=df_real,
            synthetic_data=df_synth,
            key_fields=key_fields,
            sensitive_fields=[sensitive_col],
        )
        df_scores.loc["CategoricalCAP", sensitive_col] = score
    return df_scores
