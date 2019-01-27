
import networkx as nx
import numpy as np
from struc import *

url = '../active/models/tools/edgelist_idx.txt'

with open(url, 'r') as f:
    edgelist = [[int(x) for x in y.split()] for y in f.readlines()]

edges_no_weight = [x[:2] for x in edgelist]

G = nx.Graph()
G.add_edges_from(edges_no_weight)

print ('number of edges: %d' % (len(G.nodes())))

construct_similarity_graph(G, 5, True)
