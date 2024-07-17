import pandas as pd
import numpy as np
from itertools import combinations
from src.utils import utils
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def evaluate_correlations(
    df_real: pd.DataFrame, df_synth: pd.DataFrame, list_continuous_columns: list
) -> pd.DataFrame:
    """Generates correlation scores on continuous columns

    Args:
        df_real (pd.DataFrame): Dataframe with real data
        df_synth (pd.DataFrame): Dataframe with synthetised data
        list_continuous_columns (list): Name of continuous columns

    Returns:
        pd.DataFrame: Dataframe with correlation similarity for each combination of columns
    """
    df_corr = pd.DataFrame()
    col_combinations = combinations(list_continuous_columns, 2)
    for col1, col2 in col_combinations:
        df_corr.loc[
            "correlation_score", f"{col1}_{col2}"
        ] = CorrelationSimilarity.compute(df_real[[col1, col2]], df_synth[[col1, col2]])
    return df_corr


def get_score_plot(sdv_report: QualityReport):
    """Retrieves a plotly score plot"""
    score_plot = (
        sdv_report.get_visualization(property_name="Column Shapes")
        # format xaxis
        .update_xaxes(tickfont_size=10, tickangle=-30).update_yaxes(tickfont_size=10)
        # format legend (to gain space and readibility)
        .update_layout(
            legend=dict(
                orientation="h",
                xanchor="right",
                x=1,
                yanchor="bottom",
                y=1,
            ),
        )
    )
    return score_plot


def get_correlation_plot(sdv_report: QualityReport):
    """Retrieves a plotly correlation plot"""
    corr_plot = (
        sdv_report.get_visualization(property_name="Column Pair Trends")
        # format xaxis
        .update_xaxes(tickfont_size=10, tickangle=-30).update_yaxes(tickfont_size=10)
        # format legend (to gain space and readibility)
        .update_layout(
            legend=dict(
                orientation="h",
                xanchor="right",
                x=1,
                yanchor="bottom",
                y=1,
            ),
        )
    )
    return corr_plot


def coverage(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    """Counts number of identical patients between real and synthetic dataset

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Synthetic dataset

    Returns:
        pd.DataFrame: Dataframe with counts
    """
    df_distinct = pd.DataFrame(
        {
            "number of same real patients": [
                len(df_real) - len(df_real.drop_duplicates())
            ],
            "number of same sim patients": [
                len(df_synth) - len(df_synth.drop_duplicates())
            ],
            "number of unique sim patient similar to unique real": [
                len(pd.concat([df_real.drop_duplicates(), df_synth.drop_duplicates()]))
                - len(
                    pd.concat(
                        [df_real.drop_duplicates(), df_synth.drop_duplicates()]
                    ).drop_duplicates()
                )
            ],
        }
    )

    return df_distinct


def compute_WD(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Computes Wasserstein Distance to compare two dataframes.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real dataset

    Returns:
        float: Mean of the Wasserstein Distance scores. Varies between [0,inf[. The greater the score, the more similar the data.
    """
    wd_col = []
    for col in df_real.columns:
        wd_col.append(wasserstein_distance(df_real[col], df_synth[col]))
    wd = np.mean(wd_col)
    return wd


def compute_JSD(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Computes Jensen Shannon Divergence to compare two dataframes.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real dataset

    Returns:
        float: Mean of the Jensen Shannon scores. Varies between [0, ln(2)]. The closer from 0 the more similar the data. Value can diverge if some events have a null probability of happening.
    """
    if len(df_real) != len(df_synth):
        k_sample = min(len(df_real), len(df_synth))
        df_real = df_real.sample(k_sample)
        df_synth = df_synth.sample(k_sample)
    jsd_all = jensenshannon(df_real, df_synth, axis=1)
    jsd_mean = np.mean(jsd_all)
    if jsd_mean == np.inf:
        print(
            "Events with a zero probability of happening caused infinite divergence while computing Jensen Shannon Divergence"
        )
    return jsd_mean


def compute_KSComplement(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Computes KSComplement score to compare the continuous variables distributions from two dataframes.

    Args:
         df_real (pd.DataFrame): Real dataset
         df_synth (pd.DataFrame): Dataset synthesized from real dataset

     Returns:
         float: Mean of the KSComplement scores. Varies between [0,1]. The closer to 1 the more similar the data.
    """
    continuous_columns = utils.categorize_columns(df_real)["continuous"]
    ks_all = []
    for col in df_real.columns:
        if col in continuous_columns:
            ks = KSComplement.compute(
                real_data=df_real[col], synthetic_data=df_synth[col]
            )
            ks_all.append(ks)
    ks_mean = np.mean(ks_all)
    return ks_mean


def compute_TVComplement(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Computes TVComplement score to compare the discrete variables distributions from two dataframes.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real datase

    Returns:
        float: Mean of the TVComplement scores. Varies between [0,1]. The closer to 1 the more similar the data.
    """
    discrete_columns = utils.categorize_columns(df_real)["discrete"]
    tv_all = []
    for col in df_real.columns:
        if col in discrete_columns:
            tv = TVComplement.compute(
                real_data=df_real[col], synthetic_data=df_synth[col]
            )
            tv_all.append(tv)
    tv_mean = np.mean(tv_all)
    return tv_mean


def compute_CorrelationSimilarity(
    df_real: pd.DataFrame, df_synth: pd.DataFrame
) -> float:
    """Computes a correlation score to compare correlations of continuous variables in a real dataset vs in a synthesized dataset.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real datase

    Returns:
        float: Mean of the CorrelationSimilarity scores. Varies between [0,1]. The closer to 1 the more similar the correlations between the two datasets.
    """
    continuous_columns = utils.categorize_columns(df_real)["continuous"]
    cont_col_combinations = combinations(continuous_columns, 2)
    corrnum_all = []
    for (col1, col2) in cont_col_combinations:
        corrnum = CorrelationSimilarity.compute(
            df_real[[col1, col2]], df_synth[[col1, col2]]
        )
        corrnum_all.append(corrnum)
    corrnum_mean = np.mean(corrnum_all)
    return corrnum_mean


def compute_ContingencySimilarity(
    df_real: pd.DataFrame, df_synth: pd.DataFrame
) -> float:
    """Computes a correlation score to compare correlations of discrete variables in a real dataset vs in a synthesized dataset.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real datase

    Returns:
        float: Mean of the ContingencySimilarity scores. Varies between [0,1]. The closer to 1 the more similar the correlations between the two datasets.
    """
    discrete_columns = utils.categorize_columns(df_real)["discrete"]
    disc_col_combinations = combinations(discrete_columns, 2)
    corrdisc_all = []
    for (col1, col2) in disc_col_combinations:
        corrdisc = ContingencySimilarity.compute(
            df_real[[col1, col2]], df_synth[[col1, col2]]
        )
        corrdisc_all.append(corrdisc)
    corrdisc_mean = np.mean(corrdisc_all)
    return corrdisc_mean


def evaluate_fidelity(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    list_metrics: list = [
        "WD",
        "JSD",
        "KSComplement",
        "TVComplement",
        "CorrelationSimilarity",
        "ContingencySimilarity",
    ],
) -> dict:
    """Takes the real and the synthetic datasets and compares them according to various metrics. Metrics available : WD (Wasserstein Distance), JSD (Jensen Shannon Divergence), KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity

    Args:
        df_real (pd.DataFrame): Dataset with real data
        df_synth (pd.DataFrame): Dataset with synthetic data
        list_metrics (list): List with the metrics the user wants to compute. Must be a subset of ["WD","JSD","KSComplement","TVComplement","CorrelationSimilarity", "ContingencySimilarity"]. Default is all metrics.

    Returns:
        dict: Dictionnary with the metrics names as keys and the metrics values as values.
    """
    dict_results = {}
    for metric in list_metrics:
        if metric == "WD":
            wd = compute_WD(df_real, df_synth)
            dict_results[metric] = wd

        if metric == "JSD":
            jsd = compute_JSD(df_real, df_synth)
            dict_results[metric] = jsd

        if metric == "KSComplement":
            ks = compute_KSComplement(df_real, df_synth)
            dict_results[metric] = ks

        if metric == "TVComplement":
            tv = compute_TVComplement(df_real, df_synth)
            dict_results[metric] = tv

        if metric == "CorrelationSimilarity":
            corrnum = compute_CorrelationSimilarity(df_real, df_synth)
            dict_results[metric] = corrnum

        if metric == "ContingencySimilarity":
            corrdisc = compute_ContingencySimilarity(df_real, df_synth)
            dict_results[metric] = corrdisc

    return dict_results
