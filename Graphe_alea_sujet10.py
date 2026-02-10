import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def mat_to_graph(A,n) :
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(0,n))
    for i in range (n) :
        for j in range (n) :
            if A[i][j] == 1 :
                G.add_edge(j,i)
    return G

def affiche_graph(G) :
    nx.draw(G, with_labels=True, font_weight='bold')
    return

def graph_alea(n) :
    A = np.random.randint(2, size = (n,n))
    G = mat_to_graph(A,n)
    affiche_graph(G)
    return G
    
graph_alea(5)