import networkx as nx
import matplotlib.pyplot as plt
import caffe_image as ci

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or PyDotPlus")

def get_paths(nodes, edges,start_node, end_nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    paths = []
    for end_node in end_nodes:
        paths.append(list(nx.all_simple_paths(G, start_node, end_node, 2)))
    return paths

