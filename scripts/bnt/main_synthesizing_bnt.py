import sys
import os
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.modelling.bnt.synthesizing_bnt import main_bnt_sampling
from src.loading import read_data, save_csv, load_model
from src.logger import init_logger
from src.parsers.parser_bnt import sampling_parser
import config.config as conf 

def main():
    # Initiate parser
    parser = sampling_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # loading 
    bn = load_model(conf.BUCKET_NAME, conf.PATH_MODEL, conf.FILE_BNT_MODEL)
    df_prepared = read_data(conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_PREPARED_DATA)

    # synthetising
    synth_data = main_bnt_sampling(bn, args.n_sample, df_prepared)

    # saving
    save_csv(synth_data, conf.BUCKET_NAME, conf.PATH_OUTPUT_DATA, conf.FILE_SYNTHESIZED_DATA)

if __name__ == "__main__":
    main()
    

