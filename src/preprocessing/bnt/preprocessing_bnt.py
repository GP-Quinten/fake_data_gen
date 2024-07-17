import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bamt.preprocessors import Preprocessor

def main_bnt_preprocessing(df: pd.DataFrame,
                       dict_metadata: dict) -> pd.DataFrame:
    """Prepare discrete variables in cohort data into right type
    Preprocesses data using bamt integrated preprocessor

    Args:
        df (pd.DataFrame): cohort data
        dict_metadata (dict): list of discrete variables

    Returns:
        pd.DataFrame: prepared cohort data
    """
    df_preproc = df.copy()
    
    for var in dict_metadata['columns'].keys():
        if dict_metadata['columns'][var]['sdtype'] in ['boolean', 'categorical']:
            df_preproc[var] = df_preproc[var].astype('object')
            
    # Encode discrete variables
    preprocessor = Preprocessor([
    ('encoder', LabelEncoder()),])

    df_preproc, _ = preprocessor.apply(df_preproc)

    return preprocessor, df_preproc