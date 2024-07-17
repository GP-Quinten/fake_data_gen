import cachetools
import pandas as pd
from sdv.evaluation.single_table import get_column_plot
from src.utils import sdv_utils, utils
from src.evaluating import evaluate_fidelity
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport


def evaluate_after_training(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    dict_metadata: dict,
    columns_to_plot: list,
) -> [cachetools.Cache, pd.DataFrame]:
    """Evaluates a synthetic dataset in comparison to the real one it was built from with SDV metrics.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Synthetic dataset
        dict_metadata (dict): Metadata dictionnary for SDV metrics
        columns_to_plot (list): Names of columns to plot distribution
        model (str): Model used to generate the synthetic dataset
        dataset_evaluated (str): Name of the synthetic dataset file

    Returns:
        [cachetools.Cache, pd.DataFrame]: Cache containing generated figures and dataframe with scores
    """

    sdv_report = sdv_utils.get_sdv_report(df_real, df_synth, dict_metadata)

    sdv_metadata = sdv_utils.get_metadata_from_dict(dict_metadata)

    nb_fig = len(columns_to_plot)

    if columns_to_plot == ["ALL"]:
        col_cat = utils.categorize_columns(df_real)
        columns_to_plot = col_cat["continuous"]
        nb_fig = len(columns_to_plot)

    figure_cache = cachetools.LRUCache(maxsize=nb_fig + 2)

    for col in columns_to_plot:
        fig = get_column_plot(
            real_data=df_real,
            synthetic_data=df_synth,
            column_name=col,
            metadata=sdv_metadata,
        )

        figure_cache[col] = fig

    global_score = sdv_report.get_score()
    colshapes_score = pd.DataFrame(sdv_report.get_properties()).loc[0, "Score"]
    corr_score = pd.DataFrame(sdv_report.get_properties()).loc[1, "Score"]
    scores_tab = pd.DataFrame(
        {
            "Global score": [global_score],
            "Column shapes score": [colshapes_score],
            "Correlation score": [corr_score],
        }
    )

    score_plot = (
        evaluate_fidelity.get_score_plot(sdv_report)
        # format xaxis
        .update_xaxes(
            tickfont_size=10,
            tickangle=-30,
            categoryorder="array",
            categoryarray=df_synth.columns,  # reorder columns by DAG's order
        ).update_yaxes(tickfont_size=10)
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

    figure_cache["score_plot"] = score_plot

    corrplot = (
        evaluate_fidelity.get_correlation_plot(sdv_report)
        # format xaxis
        .update_xaxes(
            tickfont_size=5,
            tickangle=-30,
            categoryorder="array",
            categoryarray=df_synth.columns,  # reorder columns by DAG's order
        ).update_yaxes(tickfont_size=5)
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

    figure_cache["corrplot"] = corrplot

    return figure_cache, scores_tab


def eval_quality_report(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, meta_data: dict
):
    """
    Takes as argument real_data, synthetic data and metadata. Returns list of objects with all info on sdv quality report (scores only)

    """
    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, meta_data)
    quality_df = quality_report.get_properties()

    scores_column_shapes = quality_report.get_details(property_name="Column Shapes")
    scores_column_pair_trends = quality_report.get_details(
        property_name="Column Pair Trends"
    )
    overall_score = quality_report.get_score()

    return (
        quality_report,
        quality_df,
        scores_column_shapes,
        scores_column_pair_trends,
        overall_score,
    )


def eval_diagnostic_report(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, meta_data: dict
):
    """
    Takes as argument real_data, synthetic data and metadata. Returns list of objects with all info on sdv diagnostic report (scores only)

    """
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(real_data, synthetic_data, meta_data)
    diagnostic_df = diagnostic_report.get_properties()

    score_synthesis = diagnostic_report.get_details(property_name="Synthesis")
    score_coverage = diagnostic_report.get_details(property_name="Coverage")
    score_bounderies = diagnostic_report.get_details(property_name="Boundaries")

    return (
        diagnostic_report,
        diagnostic_df,
        score_synthesis,
        score_coverage,
        score_bounderies,
    )


def evaluate_train_quality_and_diagnostic(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, meta_data: dict
):
    """
    Takes as argument real_data, synthetic data and metadata. Returns list of objects with all info on sdv quality and quality report (scores only)

    """
    (
        quality_report,
        quality_df,
        scores_column_shapes,
        scores_column_pair_trends,
        overall_score,
    ) = eval_quality_report(real_data, synthetic_data, meta_data)
    (
        diagnostic_report,
        diagnostic_df,
        score_synthesis,
        score_coverage,
        score_bounderies,
    ) = eval_diagnostic_report(real_data, synthetic_data, meta_data)

    return (
        quality_report,
        quality_df,
        overall_score,
        diagnostic_report,
        diagnostic_df,
    )


def get_sorted_diags_prevs_df(df: pd.DataFrame) -> tuple:
    """
    Takes a dataframe with patient_IDs as rows
    -> Returns a tuple: the list of diags in the df, and a dataframe with diags as rows and 1 column "PREVALENCE" filtered and sorted by prevalence
    """

    # keep only diags
    df_diags = df[[col for col in df.columns if col[:3] == "ICD"]]
    diag_list = list(df_diags.columns)
    df_inverted = df_diags.T

    df_inverted["PREVALENCE"] = df_inverted.sum(axis=1) / len(df) * 100
    df_prevalence_min = df_inverted.sort_values(by=["PREVALENCE"], ascending=False)

    return diag_list, df_prevalence_min.loc[:, ["PREVALENCE"]]


def compute_df_prev(df: pd.DataFrame, data_type: str):
    """
    Takes a dataframe df with only binary features (diagnoses). Computes de prevalences of all diags in df.
    Returns a dataframe with 3 columns and 1 row per diagnosis: diagnosis code, prevalence in the dataframe, and a last column indicating "real" or "synthetic" indicating the type of data considered.

    Args:
        df (pd.DataFrame): dataframe with only binary features (diagnoses)
        data_type (str): type of data : real or synthetic

    Returns:
        pd.DataFrame: Returns a dataframe with 3 columns and 1 row per diagnosis: diagnosis code, prevalence in the dataframe, and a last column indicating "real" or "synthetic" indicating the type of data considered.
    """
    ## Stats on prevalences
    diag_list, df_prev = get_sorted_diags_prevs_df(df)
    _, df_prev = utils.get_data_ready(df, df_prev, data_type)

    return diag_list, df_prev
