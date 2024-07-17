import os
import sys
import warnings
import logging
import pandas as pd

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config.config as conf
from src import loading
from src.utils import utils, sdv_utils
from src.logger import init_logger
from src.parsers.parser_evaluate import fidelity_parser
from src.evaluating import evaluate_fidelity


def main():

    # Initiate parser
    parser = fidelity_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # Loading data
    df_synth = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_SYNTH_DATA, os.path.join("synthesized_data/", args.synth_dataset)
    )
    df_real = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, args.real_dataset
    )
    logging.info("Data loaded")

    # Get columns
    df_real = df_real[df_synth.columns.to_list()]
    columns_categories = utils.categorize_columns(df_real)

    discrete_columns = columns_categories["discrete"]
    list_continuous_columns = columns_categories["continuous"]

    dict_metadata = loading.read_dict(
        conf.BUCKET_NAME, os.path.join(conf.PATH_METADATA, conf.FILE_METADATA)
    )
    sdv_report = sdv_utils.get_sdv_report(df_real, df_synth, dict_metadata)

    df_patients_coverage = evaluate_fidelity.coverage(df_real, df_synth)
    df_correlation_scores = evaluate_fidelity.evaluate_correlations(
        df_real, df_synth, list_continuous_columns
    )
    corr_plot = evaluate_fidelity.get_correlation_plot(sdv_report)
    score_plot = evaluate_fidelity.get_score_plot(sdv_report)

    dict_metrics = evaluate_fidelity.evaluate_fidelity(df_real, df_synth, conf.METRICS_TO_COMPUTE)
    df_metrics = pd.DataFrame(list(dict_metrics.items()), columns=['Metrics', 'Values'])

    if args.exp_name is not None:
        args.exp_name = args.exp_name + '_'
    else:
        args.exp_name = ''

    # saving dataframes
    for df in [df_patients_coverage, df_correlation_scores, df_metrics]:
        loading.save_csv(
            df,
            conf.BUCKET_NAME,
            os.path.join(conf.PATH_EVALUATE, "evaluate_fidelity/", "dataframes/"),
            f"{df}_{args.exp_name}{conf.TODAY}.csv"
        )

    # saving plots
    loading.save_figure_s3(
    corr_plot,
    conf.BUCKET_NAME,
    os.path.join(conf.PATH_EVALUATE, "evaluate_fidelity/", "plots/"),
    f"corr_plot_{args.exp_name}{conf.TODAY}",
    )

    loading.save_figure_s3(
        score_plot,
        conf.BUCKET_NAME,
        os.path.join(conf.PATH_EVALUATE, "evaluate_fidelity/", "plots/"),
        f"score_plot_{args.exp_name}{conf.TODAY}",
    )



if __name__ == "__main__":
    main()
