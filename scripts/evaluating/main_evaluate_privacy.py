import warnings
import logging

warnings.filterwarnings("ignore")

import config.config as conf
from src import loading
from src.logger import init_logger
from parsers.parser_evaluate import privacy_parser
from src.evaluating import evaluate_privacy


def main():

    # Initiate parser
    parser = privacy_parser()
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

    privacy_results = evaluate_privacy.privacy_check(
        df_real, df_synth, args.key_fields, args.sensitive_fields
    )


if __name__ == "__main__":
    main()
