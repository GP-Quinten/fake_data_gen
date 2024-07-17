import pandas as pd
from bamt.networks.hybrid_bn import HybridBN


def get_round(val):
    """Returns the round information (number after '.')

    Args:
        val (float): value to return round information

    Returns:
        int: number after '.' for the given float
    """
    if not '.' in str(val):
        return 0
    else:
        val = str(val).split('.')[1].rstrip('0')
        return len(val)


def main_bnt_sampling(
        bn,
        nb_patients,
        df_train=None):
    """Simulate from bayesian network virtual patients
    Post process sampled patients based in training data preprocessed (same rounding and dtypes) /!\ preprocessed data and not prepared 

    Args:
        bn (HybridBN): bayesian network
        nb_patients (int): number of patients
        df_train (pd.DataFrame): 

    Returns:
        pandas.DataFrame: simulated patients
    """
    df_simulated_patients = bn.sample(
        nb_patients)  # BUG the number of patients is lower than the one specified becase contraints are applied after generation

    # post-processing
    if df_train is not None:
        for col in df_train:
            df_simulated_patients[col] = df_simulated_patients[col].astype(
                df_train[col].dtypes)
            if df_train[col].dtype in [float, 'float64']:
                n_round = df_train[col].apply(lambda x: get_round(x)).max()
                df_simulated_patients[col] = df_simulated_patients[col].round(
                    n_round)

    return df_simulated_patients
