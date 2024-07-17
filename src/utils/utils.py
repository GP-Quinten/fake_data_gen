import pandas as pd
from sdv.metadata.single_table import SingleTableMetadata


def search_na(df):
    """
    Search for NA values in all columns
    """
    for col in df.columns:
        print(f"{col} : {df[col].isnull().values.any()}")


def get_metadata_from_dict(dict_metadata):
    metadata = SingleTableMetadata()
    for key in dict_metadata.keys():
        key = str(key)
        setattr(metadata, key, dict_metadata[key])
    return metadata


def turn_to_datetime(df: pd.DataFrame, column_name: list) -> pd.DataFrame:
    """Changes columns types to datetime.

    Args:
        df (pd.DataFrame): Pandas dataframe containing columns with types to change
        column_name (list): Names of the columns of which the type needs to be changed

    Returns:
        pd.DataFrame: Pandas dataframe
    """
    df[column_name] = df[column_name].apply(pd.to_datetime)
    return df


def compute_timegap(
    df: pd.DataFrame, upper_bound_column: str, lower_bound_column: str
) -> pd.DataFrame:
    """Computes a gap in days between two dates.

    Args:
        df (pd.DataFrame): Pandas dataframe containing at least two columns to compare
        upper_bound_column (str): Name of the upper bound column (later date)
        lower_bound_column (str): Name of the lower bound column (earlier date)

    Returns:
        pd.DataFrame: Dataframe with one column containing the timegaps.
    """
    return (
        (df[upper_bound_column].values - df[lower_bound_column].values).astype(int)
        / 8.64e13
        // 365
    ).astype(int)


def merge_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column_to_merge_on: str,
    merging_technique: str,
    df1_columns_to_keep: list,
    df2_columns_to_keep: list,
) -> pd.DataFrame:
    return (
        df1[df1_columns_to_keep]
        .merge(df2[df2_columns_to_keep], on=column_to_merge_on, how=merging_technique)
        .drop_duplicates()
    )


def categorize_columns(df: pd.DataFrame, threshold: int = 5) -> dict:
    """Categorizing columns from a dataset as discrete or continuous according to the number of modes in it.

    Args:
        df (pd.DataFrame): Dataframe
        threshold (int, optional): Number of maximum modes for a feature ti be considered discrete. Defaults to 5.

    Returns:
        dict: Keys are the category of column, values are their names. One key can have several values.
    """
    discrete_cols = []
    continuous_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if (
                len(df[col].unique()) <= threshold
            ):  # Adjust the threshold for discrete vs. continuous as needed
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]):
            discrete_cols.append(col)

    return {"discrete": discrete_cols, "continuous": continuous_cols}


def cleanup_col(
    df: pd.DataFrame, col_to_clean: str, value_to_exclude: int
) -> pd.DataFrame:
    """
    Removes patient whose value of column col_to_clean is below 'value_to_exclude'.
    Example of use : Removes df=df_demogs whose value of column age_at_last_admission is below 0.

    Args:
        df (pd.DataFrame): Pandas dataframe
        col_to_clean (str): Column to check up on
        value_to_exclude (int): To which value we want to compare the column

    Returns:
        pd.DataFrame: Cleaned pandas dataframe
    """

    # Clean up
    df = df[~(df[col_to_clean] < value_to_exclude)]
    return df


def update_dictionnary(dict: dict, list_to_match: list(), inner_dict: str):
    """Updates the content of a dictionnary to match a given list.

    Args:
        dict (dict): Dictionnary to update
        list_to_match (list): List of keys to match
        inner_dict (str): If the dict is in another one
    """
    to_remove = []
    if inner_dict is not None:
        for key in dict[inner_dict].keys():
            if key not in list_to_match:
                to_remove.append(key)
        for key in to_remove:
            del dict[inner_dict][key]
    else:
        for key in dict.keys():
            if key not in list_to_match:
                to_remove.append(key)
        for key in to_remove:
            del dict[key]
    return dict


def get_metadata_from_dict(dict_metadata: dict):
    metadata = SingleTableMetadata()
    for key in dict_metadata.keys():
        key = str(key)
        setattr(metadata, key, dict_metadata[key])
    return metadata


def get_data_ready(df1: pd.DataFrame, df2: pd.DataFrame, data_type: str) -> tuple:
    """
    Adds a column NATURE of value type = "real" or "synth" for all lines of the dataset df1 and df2 (same value).
    """
    ## Re-ordering the dataset columns
    df = df1.copy()
    df["NATURE"] = data_type
    df = df[["NATURE"] + df.columns[:-1].tolist()]
    df2["NATURE"] = data_type
    return df, df2
