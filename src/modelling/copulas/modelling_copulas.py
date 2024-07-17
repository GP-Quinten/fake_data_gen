from copulas.multivariate import GaussianMultivariate
from copulas.multivariate import VineCopula


def main_copulas_fitting(copulas_type, prepared_data, vine_type='regular', random_state=123, **kwargs):
    """_summary_

    Args:
        copulas_type (str): can be gaussian or vine
        prepared_data (DataFrame): prepared data dataframe
        vine_type (str, optional): see VineCopula documentation. Defaults to 'regular'.
        random_state (int, optional): random state. Defaults to 123.
        **kwargs:distributions

    Returns:
        copulas: fitted copulas
    """
    
    if copulas_type == 'gaussian':
        copulas = GaussianMultivariate(random_state=random_state, **kwargs)

    elif copulas_type == 'vine':
        # only gaussianKDE distributions
        copulas = VineCopula(vine_type, random_state=random_state)

    copulas.fit(prepared_data)
    return copulas
