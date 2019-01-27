import time
import networkx as nx

from struc import *

# path to edgelist with id mapped to consecutive integers.
with open('../active/models/modwalk/tmp/edgelist_idx.txt', 'r') as f:
    edgelist = [[int(x) for x in y.split()] for y in f.readlines()]


graph = nx.Graph()
for e in edgelist:
    graph.add_edge(e[0], e[1], weight=e[2])
    graph.add_edge(e[1], e[0], weight=e[2])

print('sorting degree list...')
sorted_deg_list = get_sorted_degrees(graph)
print('sorting is done.')

str_graphs = construct_similarity_graph(graph, 3, sorted_deg_list, True)
for i, edgelist in str_graphs.iteritems():
    #print(edgelist)
    with open('struc_graph_'+str(i), 'w+') as f:
        f.write('\n'.join([' '.join([str(n) for n in l]) for l in edgelist]))

