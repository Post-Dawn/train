
import numpy as np
from fastdtw import fastdtw
import heapq
import sys
import time
from helpers import update_progress
import heapq

'''
this is an implementation of structural similarity
computation in paper struc2vec.
'''


def degree_distance(a,b):
    return max(a,b)/min(a,b)-1


def k_neighbors(graph, idx, k):
    neighbors = {}
    for i in range(k):
        k_neighbors = set([idx])
        for l in range(i):
            k_neighbors = set([nbr for n in k_neighbors for nbr in graph[n]])

        neighbors[i] = k_neighbors
        k_minus = set()
        for j in range(i):
            k_minus=k_minus.union(neighbors[j])
        neighbors[i] -= k_minus

    return neighbors


def k_degrees(graph, idx, k):
    neighbors = k_neighbors(graph, idx, k)
    degrees = {i: sorted([graph.degree[v] for v in neighbors[i]]) for i in range(k)}
    return degrees


def compute_similarity(graph, u, v, k):
    degrees_u = k_degrees(graph, u, k)
    degrees_v = k_degrees(graph, v, k)
    sims = []
    for i in range(k):
        seq_u = degrees_u[i]
        seq_v = degrees_v[i]
        if len(seq_u) == 0 or len(seq_v) == 0:
            break
        sims.append(fastdtw(seq_u, seq_v, dist=degree_distance)[0])

    for i in range(1,len(sims)):
        sims[i] += sims[i-1]
    return sims


def fast_k_degrees(graph, idx, k):
    neighbors = k_neighbors(graph, idx, k)
    degrees = {}
    for i in range(k):
        degrees[i] = {}
        if i in neighbors:
            for n in neighbors[i]:
                if graph.degree(n) in degrees[i]:
                    degrees[i][graph.degree(n)] += 1
                else:
                    degrees[i][graph.degree(n)] = 1
    
    for i in degrees:
        degrees[i] = sorted([(k,v) for k,v in degrees[i].items()], key = lambda x:x[0])
    return degrees


def fast_degree_distance(a,b):
    return degree_distance(a[0],b[0]) * max(a[1],b[1])


def fast_compute_similarity(graph, u,v, k):
    degrees_u = fast_k_degrees(graph, u, k)
    degrees_v = fast_k_degrees(graph, v, k)
    sims = []
    for i in range(k):
        seq_u = degrees_u[i]
        seq_v = degrees_v[i]
        if len(seq_u) == 0 or len(seq_v) == 0:
            break
        sims.append(fastdtw(seq_u, seq_v, dist=fast_degree_distance)[0])

    for i in range(1,len(sims)):
        sims[i] += sims[i-1]
    return sims


def get_sorted_degrees(graph):
    deg_pair = sorted([(x, graph.degree(x)) for x in graph], key=lambda x:x[1])
    return deg_pair


def _binary_search(L, element, key, start, end):
    mid = int((start+end)/2)
    #print(start, end)
    if start > end:
        return -1
    if key(L[mid]) == element:
        return mid
    elif key(L[mid]) > element:
        return _binary_search(L, element, key, start, mid-1)
    else:
        return _binary_search(L, element, key, mid+1, end)


def binary_search(L, element, key=lambda x:x):
    return _binary_search(L, element, key, 0, len(L)-1)


def determine_neighbors(graph, n, k, sorted_degrees=None):
    if sorted_degrees is not None:

        deg = graph.degree(n)

        # print('vertex: ', n, ' degree:', deg)
        pos = binary_search(sorted_degrees, graph.degree(n), key=lambda x: x[1])
        assert 0 <= pos < len(sorted_degrees), "vertex degree not found in sorted degree list!"

        neighbors = []
        # print('deg sequence:', sorted_degrees)
        # print('position at:')
        # print(pos)
        left = pos - 1
        right = pos + 1
        while len(neighbors) < min(len(sorted_degrees) - 1, k):

            if left >= 0 and sorted_degrees[left][0] == n:
                left -= 1
            if right < len(sorted_degrees) and sorted_degrees[right][0] == n:
                right += 1

            # print('current search: ', left, right)
            if left < 0 and right >= len(sorted_degrees):
                break

            if left < 0:
                # cannot go left, go right
                neighbors.append(sorted_degrees[right][0])
                right += 1
            elif right >= len(sorted_degrees):
                # cannot go right, go left
                neighbors.append(sorted_degrees[left][0])
                left -= 1
            else:
                # can go both direction; choose the one which the degree difference is the smaller.
                deg_left = sorted_degrees[left][1]
                deg_right = sorted_degrees[right][1]

                if abs(deg_left - deg) < abs(deg_right - deg):
                    neighbors.append(sorted_degrees[left][0])
                    left -= 1
                else:
                    neighbors.append(sorted_degrees[right][0])
                    right += 1
        return neighbors
    else:
        deg_sequence = ([(abs(graph.degree(x) - graph.degree(n)), x) for x in graph.nodes() if x != n])
        heapq.heapify(deg_sequence)
        neighbors = []
        for _ in range(k):
            neighbors.append(heapq.heappop(deg_sequence)[1])
        return neighbors


def construct_similarity_graph(graph, k, sorted_degrees=None, print_progress=False):
    edgelists = {}
    for i in range(k):
        edgelists[i]=[]

    traversed = set()
    num_total = len(graph.nodes())
    start = time.time()
    num_per_node = int(np.log(len(graph)))

    for idx, n in enumerate(graph.nodes()):
        deg = graph.degree(n)
        nbrs = determine_neighbors(graph, n, num_per_node, sorted_degrees)
        for nbr in nbrs:
            sims = fast_compute_similarity(graph, n, nbr, k)
            for j in range(len(sims)):
                if (n,nbr) not in traversed and (nbr, n) not in traversed:
                    edgelists[j].append([n,nbr,np.exp(-sims[j])])

            traversed.add((n,nbr))

        if print_progress:
            update_progress(idx, num_total, start)

    return edgelists
