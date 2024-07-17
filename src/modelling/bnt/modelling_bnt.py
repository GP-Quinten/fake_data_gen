import pandas as pd
import logging
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from bamt.preprocessors import Preprocessor
from bamt.networks.hybrid_bn import HybridBN

from src.visualizing.bnt import visualizing_bnt


def learn_structure_bnt(df: pd.DataFrame,
                        preprocessor: Preprocessor,
                        df_constraints: pd.DataFrame,
                        scoring_function: str) -> HybridBN:
    """Learns the structure of a hybrid bayesian network from data

    Args:
        df (pd.DataFrame):  cohort data 1 row / patient & 1 column / variable
        preprocessor (Preprocessor): object containing information on daa variables
        df_constraints (pd.DataFrame): 1 row / constraint & 1 column / parent - child
        scoring_function (str): scoring function to maximize (BIC, AIC, K2)

    Returns:
        HybridBN: bayesian network
    """
    bn = HybridBN(has_logit=True,  # to allow link between discrete and continuous variables
                  use_mixture=True)

    # Learn structure of the BN
    bn.add_nodes(preprocessor.info)

    # structure learning , optimizer ='HC' (HillClimbing) or Evo
    params = get_blacklist_dict(df=df_constraints)

    bn.add_edges(df,
                 scoring_function=(scoring_function,),
                 params=params)

    return bn


def get_blacklist_dict(df: pd.DataFrame) -> dict:
    """Retrieves a dictionary of parameters to remove edges given a dataframe of constraints"""
    # embed parent and child columns into a tuple
    df_copy = df.copy()
    df_copy["contraint_tuple"] = list(zip(df_copy.parent, df_copy.child))

    params = {}
    params["bl_add"] = df_copy.contraint_tuple.to_list()

    return params


def get_outdict(graph) -> dict:
    """
    Get a json file to save graph
    """
    new_weights = {str(key): graph.weights[key] for key in graph.weights}
    outdict = {
        'has_logit': graph.has_logit,
        'use_mixture': graph.use_mixture,
        'info': graph.descriptor,
        'edges': graph.edges,
        'parameters': graph.distributions,
        'weights': new_weights
    }
    return outdict


def main_bnt_fitting(df: pd.DataFrame,
                     preprocessor: Preprocessor,
                     df_constraints: pd.DataFrame,
                     scoring_function: str,
                     plot_path='bnt_plot.html'):
    """Learns the structure of a hybrid bayesian network from data

    Args:
        df (pd.DataFrame):  cohort data 1 row / patient & 1 column / variable
        preprocessor (Preprocessor): object containing information on daa variables
        df_constraints (pd.DataFrame): 1 row / constraint & 1 column / parent - child
        scoring_function (str): scoring function to maximize (BIC, AIC, K2)

    Returns:
        HybridBN: bayesian network fited
    """

    # Learn structure
    logging.info("-----Learning of bayesian network from PD cohort-----")
    bn = learn_structure_bnt(df,
                             preprocessor,
                             df_constraints,
                             scoring_function)
    df_bn_info = bn.get_info()

    # learn parameters of the BN
    logging.info("-----Fitting bayesian network's parameters-----")
    bn.fit_parameters(df)

    # plot network
    logging.info("-----Plotting bayesian network-----")
    bnt_visualization.plot_bn(bn, nb_events=2, save_path=plot_path)

    return bn
