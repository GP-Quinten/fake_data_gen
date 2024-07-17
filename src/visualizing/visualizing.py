import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cachetools
from sdmetrics.reports import utils as sdmetrics_utils
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport


def concat_1d_cc(
    data_real, data_synth, fig=None, title=None, position=None, bins=20, label=None
):
    """Plot 1 dimensional data in a histogram."""
    fig = fig or plt.figure()
    position = position or 111

    ax = fig.add_subplot(position)
    ax.hist(data_real, density=True, bins=bins, alpha=0.8, label="Real")
    ax.hist(data_synth, density=True, bins=bins, alpha=0.8, label="Synthetic")

    if label:
        ax.legend()

    if title:
        ax.set_title(title)
        ax.title.set_position([0.5, 1.05])

    return ax


def hist_compare_real_synth(real, synth, columns=None, figsize=None):
    """Generate a 1d scatter plot comparing real/synthetic data.
    Warning: do not support more than 6 features.

    Args:
        real (pd.DataFrame):
            The real data.
        synth (pd.DataFrame):
            The synthetic data.
        columns (list):
            The name of the columns to plot.
        figsize:
            Figure size, passed to matplotlib.
    """
    if len(real.shape) == 1:
        real = pd.DataFrame({"": real})
        synth = pd.DataFrame({"": synth})

    if columns is None:
        columns = real.columns

    num_cols = len(columns)
    fig_cols = min(2, num_cols)
    fig_rows = (num_cols // fig_cols) + 1
    prefix = f"{fig_rows}{fig_cols}"

    figsize = figsize or (5 * fig_cols, 3 * fig_rows)
    fig = plt.figure(figsize=figsize)

    for idx, column in enumerate(columns):
        position = int(prefix + str(idx + 1))
        concat_1d_cc(
            real[column], synth[column], fig=fig, position=position, title=column
        )

    plt.tight_layout()


def plot_losses(d_loss: list, g_loss: list):
    """Graph of discriminator and generator loss"""
    # plot loss
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.plot(d_loss, label="discriminator")
    plt.plot(g_loss, label="generator")
    return fig


def plot_diags_prev(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
):
    """
    Takes the 2 dataframes of the prevalences of all diags of real and synthetic data, and makes a histogram plot to compare the prevalences

    """
    df = pd.concat([df1, df2]).rename_axis("ICD_codes").reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.barplot(y="PREVALENCE", x="ICD_codes", data=df, hue="NATURE")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Diag top Prevalences", fontsize=12)
    ax.set_xlabel("ICD codes")
    ax.set_ylabel("Prevalence (%)", fontsize=12)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.legend()

    return fig


def plotting_distrib_and_correlations(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    meta_data: dict,
    numerical_columns: list,
):
    """
    Takes real data, syntheic data and metadata as arguments.
    Get the graphs of all distribution plots (real vs synthetic data) of sdv reportand store them in a cache that is returned.
    Get the graphs of all correlation plots (real vs synthetic data) of sdv report and store them in a cache that is returned.
    Returns a tuple with both caches.

    """
    nb_fig_1 = len(real_data.columns)
    nb_fig_2 = len(numerical_columns) ** 2
    figure_cache_column_plot = cachetools.LRUCache(maxsize=nb_fig_1)
    figure_cache_column_pair_plot = cachetools.LRUCache(maxsize=nb_fig_2)

    for col in real_data.columns:
        fig = sdmetrics_utils.get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_name=col,
            metadata=meta_data,
        )
        figure_cache_column_plot[col] = fig

    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            fig = sdmetrics_utils.get_column_pair_plot(
                real_data=real_data,
                synthetic_data=synthetic_data,
                column_names=[numerical_columns[i], numerical_columns[j]],
                metadata=meta_data,
            )
            figure_cache_column_pair_plot[
                f"{numerical_columns[i]}_{ numerical_columns[j]}"
            ] = fig

    return figure_cache_column_plot, figure_cache_column_pair_plot


def viz_reports(
    quality_report: QualityReport, diagnostic_report: DiagnosticReport, cols: list
):
    """
    Takes quality and diagnostic reports in arguments. Returns all the graphs related to those reports and store them in a cache that is returned.

    Args:
        quality_report (QualityReport): Sdv Quality generated report
        diagnostic_report (DiagnosticReport): Sdv Diagnostic generated report
        cols (list): list of cols with order

    Returns:
        _type_: _description_
    """

    nb_fig = 5
    figure_cache = cachetools.LRUCache(maxsize=nb_fig)

    fig_column_shapes = (
        quality_report.get_visualization(property_name="Column Shapes")
        # format xaxis
        .update_xaxes(
            tickfont_size=5,
            tickangle=-30,
            categoryorder="array",
            categoryarray=cols,  # reorder columns by DAG's order
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

    figure_cache["Column Shapes"] = fig_column_shapes

    fig_column_pair_trends = (
        quality_report.get_visualization(property_name="Column Pair Trends")
        # format xaxis
        .update_xaxes(
            tickfont_size=5,
            tickangle=-30,
            categoryorder="array",
            categoryarray=cols,  # reorder columns by DAG's order
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
    figure_cache["Column Pair Trends"] = fig_column_pair_trends

    fig_synthesis = diagnostic_report.get_visualization(property_name="Synthesis")
    figure_cache["Synthesis"] = fig_synthesis

    fig_coverage = (
        diagnostic_report.get_visualization(property_name="Coverage")
        # format xaxis
        .update_xaxes(
            tickfont_size=5,
            tickangle=-30,
            categoryorder="array",
            categoryarray=cols,  # reorder columns by DAG's order
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
    figure_cache["Coverage"] = fig_coverage

    fig_bounderies = (
        diagnostic_report.get_visualization(property_name="Boundaries")
        # format xaxis
        .update_xaxes(
            tickfont_size=5,
            tickangle=-30,
            categoryorder="array",
            categoryarray=cols,  # reorder columns by DAG's order
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
    figure_cache["Bounderies"] = fig_bounderies

    return figure_cache
