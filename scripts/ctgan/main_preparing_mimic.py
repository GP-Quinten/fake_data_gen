import os
import sys
import warnings
import json

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import logging
import config.config as conf
import config.config_mimic_iii as conf_mimic
from src.logger import init_logger
from src.parsers import parser_preparing
from src import loading
from src.utils import sdv_utils, utils
from src.preparing.MIMIC_III.preparing import preparing


def main():
    # Initiate parser
    parser = parser_preparing.parser_preparing()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # Load data
    logging.info("Loading data")
    df_patients = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf_mimic.FILENAME_PATIENTS
    )
    df_adm = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf_mimic.FILENAME_ADMISSIONS
    )
    df_diag = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf_mimic.FILENAME_DIAGNOSES
    )
    df_labs = loading.read_data(conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf_mimic.FILENAME_LABS)
    df_labs_desc = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf_mimic.FILENAME_LABS_DESC
    )
    logging.info("Data tables loaded")

    real_data = preparing(
        df_patients=df_patients,
        df_adm=df_adm,
        df_diag=df_diag,
        df_labs=df_labs,
        df_labs_desc=df_labs_desc,
        patient_id="SUBJECT_ID",
        col_date_of_death="DOD",
        col_date_of_birth="DOB",
        col_deceased_indicator="EXPIRE_FLAG",
        col_admission_time="ADMITTIME",
        col_ethnicity="ETHNICITY",
        col_to_keep_patients=conf_mimic.COL_TO_KEEP_PATIENTS,
        col_to_keep_adm=conf_mimic.COL_TO_KEEP_ADMISSION,
        col_to_keep_demog=conf_mimic.COL_TO_KEEP_DEMOG,
        col_diag="ICD9_CODE",
        min_prev=conf_mimic.MIN_PREV,
        loinc_map=conf_mimic.LABELS_TO_LOINC,
        col_loinc_codes="LOINC_CODE",
        col_lab_id="ITEMID",
        col_label="LABEL",
        col_num_values="VALUENUM",
        remove_nan=args.remove_nan,
    )

    logging.info("Creating metadata dict")

    real_data = real_data.drop("SUBJECT_ID", axis=1)
    columns_categories = utils.categorize_columns(real_data)
    discrete_columns = columns_categories["discrete"]

    metadata = sdv_utils.build_dict_metadata_from_df(
        real_data, "SUBJECT_ID", discrete_columns
    )

    loading.save_dict(
        metadata, conf.BUCKET_NAME, conf.PATH_METADATA, conf.FILE_METADATA
    )

    logging.info("Metadata dict saved")

    loading.save_csv(
        real_data,
        conf.BUCKET_NAME, 
        conf.PATH_OUTPUT_DATA, 
        conf.FILE_PREPARED_DATA)

    logging.info("Real data saved")


if __name__ == "__main__":
    main()
