
import sys
import os

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.modelling.copulas.synthesizing_copulas import main_copulas_sampling
from src.preprocessing.copulas.preprocessing_copulas import main_copulas_preprocessing
from src.loading import read_data, save_csv, load_model, read_dict
from src.logger import init_logger
from src.parsers.parser_copulas import sampling_parser
import config.config as conf 

def main():
    # Initiate parser
    parser = sampling_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # loading 
    copulas = load_model(conf.BUCKET_NAME, conf.PATH_MODEL, conf.FILE_COPULAS_MODEL)
    df_prepared = read_data(conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPARED_DATA)
    metadata = read_dict(
        conf.BUCKET_NAME, os.path.join(conf.PATH_METADATA, args.metadata_file)
    )

    # preprocessing 
    df_preproc, metadata, df_processor = main_copulas_preprocessing(df_prepared, metadata)

    # synthesising
    synth_data = main_copulas_sampling(copulas, args.n_sample, df_processor)

    # saving
    save_csv(synth_data, conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_SYNTHESIZED_DATA)

if __name__ == "__main__":
    main()
    