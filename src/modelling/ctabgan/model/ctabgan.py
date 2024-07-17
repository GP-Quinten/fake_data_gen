import logging
import time

# Used for pre/post-processing of the input/generated data
from src.modelling.ctabgan.model.pipeline import data_preparation

# Model class for the CTABGANSynthesizer
from src.modelling.ctabgan.model.synthesizer import ctabgan_synthesizer


class CTABGAN:

    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution
    6) integer_columns -> list of numeric column names without floating numbers
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data
    3) generate_samples() -> returns a generated and post-processed sythetic dataframe with the same size and format as per the input data

    """

    def __init__(
        self,
        df_real_data,
        categorical_columns=[],
        log_columns=[],
        mixed_columns={},
        integer_columns=[],
        problem_type={},
        test_ratio=0.20,
        epochs=10,
        batch_size=100,
        lr=2e-4,
        l2scale=1e-5,
    ):
        self.__name__ = "CTABGAN"

        self.synthesizer = ctabgan_synthesizer.CTABGANSynthesizer(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            l2scale=l2scale,
        )
        self.df_real_data = df_real_data
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):
        self.data_prep = data_preparation.DataPrep(
            self.df_real_data,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio,
        )

        logging.info("Dataprep of ctabgan repo made")

        start_time = time.time()
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            pb_type=self.problem_type,
        )
        end_time = time.time()
        fitting_time = str((end_time - start_time) // 60)
        logging.info("Finished training in " + fitting_time + " minutes.")

    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.df_real_data))
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
