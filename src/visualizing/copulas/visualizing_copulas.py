from copulas.visualization import hist_1d, side_by_side
import matplotlib.pyplot as plt
import pandas as pd

def univ_distrib(univariates, data):
    """
    Visualize original distributions and uniform distributions 
    Inputs:
        univariates (copula.univariates): univariates distributions
        data (pd.DataFrame): original dataframe
    """
    cols = data.columns
    for i in range(len(cols)):
        dict_distrib = {}
        col = cols[i]
        dict_distrib[col]=data[col]
        dict_distrib['{} transformed with {}'.format(col,univariates[i].to_dict()['type'])]=univariates[i].cdf(data[col])
        side_by_side(hist_1d, dict_distrib)

    