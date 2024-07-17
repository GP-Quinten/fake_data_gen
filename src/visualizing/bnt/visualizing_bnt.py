import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import norm
import networkx as nx
import pyvis as pv


def plot_bn(bn, nb_events: int, save_path: str):
    """Generate the plot of a bayesian network and save it locally as .html.

    Args:
        bn (HybridBN): bayesian network
        ref (dict): feature referential
        nb_events (int): number of visits
        save_path (str): path for output file
    """

    # CONSTANTS
    VISIT_TO_COLOR = {
        -1: '#bae1ff', 
        0: '#baffc9', 
        1: '#ffffba', 
        2: '#ffdfba', 
        3: '#ffb3ba', 
    }
    NODE_SIZE = 50
    NODE_FONTSIZE = 35
    SCALE_X, SCALE_Y = 3_200, 1_800
    NODE_BORDERWIDTH_WITH_CHILDREN, NODE_BORDERWIDTH_WITHOUT_CHILDREN = 3, 0
    NODE_BORDERWIDTHSELECTED_WITH_CHILDREN, NODE_BORDERWIDTHSELECTED_WITHOUT_CHILDREN = 5, 1
    EDGE_SELECTIONWIDTH = 7.5
    EDGE_SMOOTH ={'type': 'cubicBezier', 'roundness': 0.75}

    # initialize nodes & edges informations
    nodes_infos = dict()
    edges_infos = dict()
    layers_infos = dict()

    # populate nodes & edges informations
    for node in bn.nodes:
        nodes_infos[node.name] = {
            'label': node.name,
            'title': f'{node.name} ({node.type})'.replace('\n', ' '), 
            'size': NODE_SIZE, 
            'font': {'size': NODE_FONTSIZE}, 
        }
    for edge in bn.edges:
        edge = tuple(edge)
        edges_infos[edge] = {
            'selectionWidth': EDGE_SELECTIONWIDTH, 
            'smooth': EDGE_SMOOTH, 
        }

    # populate nodes & edges informations from networkx graph
    ## xy coordinates from multipartite layout
    ## borderWidth and borderWidthSelected based on children or not
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(nodes_infos.keys())
    nx_graph.add_edges_from(edges_infos.keys())
    for layer, nodes in enumerate(nx.topological_generations(nx_graph)):
        layers_infos[layer] = len(nodes)
        for node in nodes:
            nx_graph.nodes[node]['layer'] = layer
            nodes_infos[node]['layer'] = layer
    for node, xy in nx.multipartite_layout(nx_graph, subset_key='layer', align='vertical').items():
        nodes_infos[node]['x'] = xy[0] * SCALE_X
        nodes_infos[node]['y'] = xy[1] * SCALE_Y / (layers_infos[nodes_infos[node]['layer']] / np.max(list(layers_infos.values())))**(3/4)
        nodes_infos[node]['borderWidth'] = NODE_BORDERWIDTH_WITHOUT_CHILDREN if (nx_graph.out_degree(node)==0) else NODE_BORDERWIDTH_WITH_CHILDREN
        nodes_infos[node]['borderWidthSelected'] = NODE_BORDERWIDTHSELECTED_WITHOUT_CHILDREN if (nx_graph.out_degree(node)==0) else NODE_BORDERWIDTHSELECTED_WITH_CHILDREN

    # create pyvis graph
    pv_graph = pv.network.Network(
        directed=True, 
        neighborhood_highlight=True, 
        #heading='Learned Bayesian Network', #BUG display twice
        notebook=True, 
    )
    for node, node_infos in nodes_infos.items():
        pv_graph.add_node(
            node, **node_infos, 
        )
    for edge, edge_infos in edges_infos.items():
        pv_graph.add_edge(
            *edge, **edge_infos, 
        )
    pv_graph.toggle_physics(False)

    # save pyvis graph
    pv_graph.show(save_path) # TO DO: save html into s3 bucket
