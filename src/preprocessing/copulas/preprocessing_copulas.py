from src import utils
from sdv.data_processing.data_processor import DataProcessor

def main_copulas_preprocessing(df_ppmi, dict_metadata, enforce_rounding=True, enforce_min_max_values=True):
    """
    Input:
        df_ppmi (pd.DataFrame): input dataframe to prepare
        dict_metadata (dict): metadata needed by sdv
        enforce_rounding (bool):
            Define rounding scheme for FloatFormatter. If True, the data returned by
            reverse_transform will be rounded to that place. Defaults to True.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by reverse_transform of the numerical
            transformer, FloatFormatter, to the min and max values seen during fit.
            Defaults to True.
    Returns:
        (pd.DataFrame): prepared data
        (metadata): metadata object constructed from dict
        (sdv.data_processing.data_processor): data processor fitted on data
    """
    # init metadata
    metadata = utils.get_metadata_from_dict(dict_metadata)

    #prepare data
    data_processor = DataProcessor(metadata, 
                                   enforce_rounding=enforce_rounding, 
                                   enforce_min_max_values=enforce_min_max_values)
    data_processor.fit(df_ppmi)
    df_ppmi_prep = data_processor.transform(df_ppmi)
    return df_ppmi_prep, metadata, data_processor