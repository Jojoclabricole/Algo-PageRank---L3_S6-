import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

alpha = 0.8
epsilon = 1e-6


def mat_to_graph(A,n) :
    """
    Convertit une matrice d'adjacence A en graphe orienté NetworkX.

    Complexité :
    O(n^2) car on parcourt toute la matrice d'adjacence.
    """

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n) :
        for j in range(n) :
            if A[i][j] == 1 :
                G.add_edge(j,i)

    return G


def affiche_graph(G) :
    """
    Affiche le graphe.

    Complexité : Complexité de la fonction draw : couteuse pour des grandes valeurs de n.
    """

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    print(G)
    return


def graph_alea(n) :
    """
    Génère une matrice d'adjacence aléatoire.

    Complexité :
    O(n^2) pour générer la matrice.
    """

    A = np.random.randint(2, size = (n,n))
    
    #Uniquement pour les petits graphes, sinon ça prend trop de temps à afficher.
    #G = mat_to_graph(A,n)
    #affiche_graph(G)

    return A


def calcul_P(A,n) :
    """
    Calcule la matrice de transition P du graphe.

    P[i,j] = probabilité de passer de j vers i.

    Complexité :
    O(n^2)
    """

    P = np.zeros((n,n))

    
    for j in range(n) :

        # Calcul du degré sortant de j
        deg = np.sum(A[:,j])
     
        # remplissage de la colonne j de P
        if deg != 0 :
            for i in range(n) :
                
                if A[i][j] == 1:
                    P[i][j] = 1 / deg

    return P


def calcul_G(A,n) :
    """
    Construit la matrice de Google :

        G = αP + (1-α)/n * U

    où U est la matrice de téléportation.

    Complexité :
    O(n^2)
    """

    P = calcul_P(A,n)
    U = np.ones([n,n])

    G = alpha*P + 1/n * (1 - alpha) * U

    return G


def surfer_alea(n):
    """
    Génère un vecteur de probabilité aléatoire
    représentant la position initiale du surfer.

    Complexité :
    O(n)
    """

    x = np.random.rand(n)
    x = x / x.sum()

    return x


def pagerank(G,n) :
    """
    Calcule le vecteur PageRank par itération :

        x_{k+1} = G x_k

    jusqu'à convergence.

    Critère d'arrêt :
        max |x_{k+1} - x_k| < epsilon : toutes les composantes du vecteur de probabilité sont proches de leur valeur limite.

    Complexité :
    O(k * n^2)
    où k est le nombre d'itérations nécessaires
    pour atteindre la convergence.
    """

    x = surfer_alea(n)

    xnext = np.dot(G,x)
    diff = np.max(np.abs(x - xnext))

    while diff > epsilon :

      x = xnext
      xnext = np.dot(G,x)

      diff = np.max(x - xnext)

    return x


# Tests et programme principal

n = 1000

A = graph_alea(n)
print("Matrice d'adjacence :")
print(A)

G = calcul_G(A,n)

proba = pagerank(G,n)

print("\nVecteur PageRank :")
print(proba)

print("\nSomme :", proba.sum())
