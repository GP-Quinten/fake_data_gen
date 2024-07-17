import os
import sys

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.logger import init_logger
from src.parsers.parser_copulas import simple_parser
from src.loading import read_data, save_csv, save_model
from src.preprocessing.copulas.preprocessing_copulas import main_copulas_preprocessing
from src.modelling.copulas.modelling_copulas import main_copulas_fitting

import config.config as conf 

def main():

    # Initiate parser
    parser = simple_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # load data
    df_prepared = read_data(conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPARED_DATA)

    # preprocessing 
    df_preproc, metadata, df_processor = main_copulas_preprocessing(df_prepared, conf.METADATA)
    save_csv(df_preproc, conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPROC_DATA)
    
    # modelling
    copula_type=conf.COPULA_TYPE #can be 'vine'

    vine_type=None #only if vine, can be ‘center’,’direct’,’regular’
    if copula_type == 'vine':
        vine_type=conf.VINE_TYPE #only if vine, can be ‘center’,’direct’,’regular’

    gaussian_copulas = main_copulas_fitting(copula_type, df_preproc, vine_type=vine_type, random_state=conf.RANDOM_STATE)
    
    # save model
    save_model(gaussian_copulas, conf.BUCKET_NAME, conf.PATH_MODEL, conf.FILE_COPULAS_MODEL)
    
if __name__ == "__main__":
    main()
    