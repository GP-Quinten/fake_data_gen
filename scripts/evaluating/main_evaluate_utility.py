import warnings
import logging
import pandas as pd

warnings.filterwarnings("ignore")

import config.config as conf
from src import loading
from src.logger import init_logger
from parsers.parser_evaluate import utility_parser
from src.evaluating import evaluate_utility


def main():
    # Initiate parser
    parser = utility_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # Loading data
    df_synth = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_SYNTH_DATA, args.synth_dataset
    )
    df_real = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_REAL_DATA, args.real_dataset
    )
    logging.info("Data loaded")

    # Get columns
    df_real = df_real[df_synth.columns.to_list()]

    best_perf, df_cv_results = evaluate_utility.discriminate_synth_data(df_real=df_real, 
                                                                             df_synth=df_synth, 
                                                                             models_params=conf.MODELS_PARAMS, 
                                                                             perf_metrics=conf.PERF_METRICS, 
                                                                             best_perf_metric=conf.BEST_PERF_METRIC, 
                                                                             n_cv=conf.N_CV, 
                                                                             split_perf=conf.SPLIT_PERF, 
                                                                             cv_random_state=conf.RANDOM_STATE, 
                                                                             shap=conf.SHAP)
    return best_perf, df_cv_results

if __name__ == "__main__":
    main()
