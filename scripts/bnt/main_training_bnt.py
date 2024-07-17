import os
import sys

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config.config as conf 
from src.logger import init_logger
from src.parsers.parser_bnt import simple_parser
from src.loading import read_data, save_csv, save_model
from src.preprocessing.bnt.preprocessing_bnt import main_bnt_preprocessing
from src.modelling.bnt.modelling_bnt import main_bnt_fitting


def main():

    # Initiate parser
    parser = simple_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # load data
    df_prepared = read_data(conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPARED_DATA)
    df_constraints = read_data(conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf.FILE_BN_CONSTRAINTS)

    # preprocessing 
    preprocessor, df_preproc = main_bnt_preprocessing(df_prepared, conf.METADATA)
    save_csv(df_preproc, conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPROC_DATA)
    
    # modelling
    bn = main_bnt_fitting(df=df_preproc,
                                preprocessor=preprocessor,
                                df_constraints=df_constraints,
                                scoring_function=conf.SCORING_FUNC)
    # plot_path=os.path.join('s3://', conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, 'bnt_plot{}.html'.format(conf.DATABASE)))
    # TO DO: Save html viz into s3 bucket
    
    # save model
    save_model(bn, conf.BUCKET_NAME, conf.PATH_MODEL, conf.FILE_BNT_MODEL)

if __name__ == "__main__":
    main()
    