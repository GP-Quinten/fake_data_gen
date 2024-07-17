import sys
import pandas as pd
import os
import logging

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.parsers.parser_bnt import simple_parser
from src.loading import read_data, save_csv
from src.logger import init_logger
from src.utils import utils, sdv_utils
from src import loading
from src.preparing.ppmi.preparing import main_ppmi_preparing
import config.config_ppmi as conf_ppmi
import config.config as conf 

def main():
    # Initiate parser
    parser = simple_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # loading and preprocessing
    df_ppmi = read_data(conf_ppmi.BUCKET_NAME, conf_ppmi.PATH_RAW_DATA,conf_ppmi.FILE_PPMI_RAW_DATA)
    df_ppmi_prepared = main_ppmi_preparing(df_ppmi=df_ppmi, 
                                            columns=conf_ppmi.COL_LIST,
                                            drop_na=True)
    
    # save 
    save_csv(df_ppmi_prepared, conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPARED_DATA)

    logging.info("Creating metadata dict")

    columns_categories = utils.categorize_columns(df_ppmi_prepared)
    discrete_columns = columns_categories["discrete"]

    metadata = sdv_utils.build_dict_metadata_from_df(
        df_ppmi_prepared, "SUBJECT_ID", discrete_columns
    )

    loading.save_dict(
        metadata, conf.BUCKET_NAME, conf.PATH_METADATA, conf.FILE_METADATA
    )

    logging.info("Metadata dict saved")

if __name__ == "__main__":
    main()
    