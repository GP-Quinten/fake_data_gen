import pandas as pd
from itertools import product, permutations
from typing import List, Tuple


def format_constraints(df: pd.DataFrame) -> dict:
    """Retrieve dictionary with as keys tuple of modality constraint and value type of constraint for e.g. 'blacklist' or whitelist"""
    constraints_by_modality = (
        df
        .set_index('Unnamed: 0')
        .rename_axis('parent')
        .rename_axis('child', axis=1)
        .stack('child')
        .rename('constraint')
        .to_dict()
    )
    return constraints_by_modality

def get_nodes(vars_static: list, vars_dynamic: list, nb_events) -> List[Tuple]:
    """Retrieves a list of tuple of name of node and an int corresponding either to -1 if static
      or to the number of visit if dynamic"""
    nodes = [
    *product(vars_static, [-1]), 
    *product(vars_dynamic, range(nb_events)),]
    return nodes

def get_node_constraints(nodes,
                         ref: dict,
                         constraints_by_modality) -> pd.DataFrame:

    constraints_by_node = list()

    # iterate over possible edges
    for node_from, node_to in permutations(nodes, 2):

        # check for constraint on times
        time_constraint = check_time_constraint(node_from, node_to)
        if time_constraint is not None:
            constraints_by_node.append([
                format_node_name(node_from), 
                format_node_name(node_to), 
                time_constraint, 
            ])
            continue

        # check for constraint on modalities
        modality_constraint = check_modality_constraint(ref=ref,
                                                        constraints_by_modality=constraints_by_modality,
                                                        node_from=node_from, node_to=node_to)
        if modality_constraint is not None:
            constraints_by_node.append([
                format_node_name(node_from), 
                format_node_name(node_to), 
                modality_constraint, 
            ])
            continue

    df = pd.DataFrame(constraints_by_node, columns=['parent', 'child', 'constraint'])
    return df

def check_time_constraint(node_from, node_to):
    if node_from[1] > node_to[1]:
        return 'blacklist'

def check_modality_constraint(ref: dict,
                              constraints_by_modality,
                              node_from: str, node_to: str):
    modality_from = ref.get(node_from[0]).get('variable_modality')
    modality_to = ref.get(node_to[0]).get('variable_modality')
    modality_constraint = constraints_by_modality.get((modality_from, modality_to))
    return modality_constraint

def format_node_name(node):
    if node[1] == -1:
        return node[0]
    else:
        return f'{node[0]}_{node[1]:.0f}'
