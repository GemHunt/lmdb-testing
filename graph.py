import networkx as nx
import matplotlib.pyplot as plt

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

def get_paths(nodes, edges):
    print 'Starting Network build'

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print 'Network built'

    paths = []
    #paths.append(list(nx.all_simple_paths(G,6280,3664,2)))
    paths.append(list(nx.all_simple_paths(G,9813,6111,3)))
    return paths
