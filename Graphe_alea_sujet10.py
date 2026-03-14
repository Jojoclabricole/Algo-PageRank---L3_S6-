import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

alpha = 1
epsilon = 1e-6


def mat_to_graph(A,n) :
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n) :
        for j in range(n) :
            if A[i][j] == 1 :
                G.add_edge(j,i)

    return G
# Compléxité : O(n^2) pour parcourir la matrice d'adjacence et ajouter les arêtes au graphe.


def affiche_graph(G) :
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    print(G)
    return
# Compléxité : O(n + m) pour dessiner le graphe, où n est le nombre de nœuds et m est le nombre d'arêtes. au pire des cas, m = n^2, donc la complexité en O(n^2) dans le pire des cas.


def graph_alea(n) :
    A = np.random.randint(2, size = (n,n))
    #G = mat_to_graph(A,n)
    #affiche_graph(G) #Uniquement pour les petits graphes, sinon ça prend trop de temps à afficher.
    return A
# Compléxité : O(n^2) pour générer la matrice, O(n^2) pour la convertir en graphe, et inconnu pour afficher le graphe mais très élevé. Donc 2*O(n^2) au total sans affichage, même complexité que pour l'affichage sinon.


def calcul_P(A,n) :

    P = np.zeros((n,n))
    deg = np.zeros(n)

    # Calcul du degré de chaque noeud
    for i in range(n) :
        s = 0
        for j in range(n) :
            if A[i][j] == 1 :
                s += 1
        deg[i] = s

    # Calcul de la matrice P
    for i in range(n) : 
        for j in range(n) :
            if A[i][j] == 1 and deg[j] != 0 :
                P[i][j] = 1 / deg[j]

    return P
# Compléxité : O(n^2) pour calculer les degrés, O(n^2) pour calculer la matrice P. Donc 2*O(n^2).


def calcul_G(A,n) :
    P = calcul_P(A,n)
    U = np.ones([n,n])

    G = alpha*P + 1/n * (1 - alpha) * U

    print(G)
    return G
# Compléxité : 2*O(n^2) pour calculer P, O(n^2) pour calculer G, donc 3*O(n^2) dans le pire des cas.

def limite_G(G,n) :

    Glim = copy.deepcopy(G)
    Gnext = np.dot(G,G)

    diff = np.abs(Glim - Gnext)

    # Calcul de la limite de G en itérant jusqu'à ce que la différence entre deux itérations soit inférieure à epsilon. Potentiel bug à vérifier ici
    while np.all(diff > epsilon) :

      Glim = Gnext
      Gnext = np.dot(Gnext,Gnext) # Calcul de Gnext^2 au lieu de Gnext*G pour améliorer la complexité de l'algorithme
      diff = np.abs(Glim - Gnext)

    print(Glim)
    return Glim
# Compléxité : O(n^3 * log(k)) avec O(n^3) pour le produit matriciel et k le nombre d'itérations nécessaires pour atteindre la convergence. En utilisant la méthode de l'exponentiation rapide, on peut réduire la complexité à O(n^3 log(k)).)


def surfer_alea(n):
    x = np.random.rand(n)   # nombres aléatoires entre 0 et 1
    x = x / x.sum()         # normalisation pour que la somme soit égale à 1
    return x
# Compléxité : 2*O(n) pour générer les nombres aléatoires et normaliser le vecteur.




# Test et programme principal

A = graph_alea(100)
print(A)

G = calcul_G(A,100)

proba = np.dot(limite_G(G,100),surfer_alea(100))
# En temps n^2 + n^3 log(k) + 2n donc en temps n^3 log(k), donc en temps 1000000000 * log(k) environ, avec k le nombre d'itérations nécessaires pour atteindre la convergence de la limite de G.

print(proba.sum()) #problème, la somme n'est pas égale à 1, mais s'en rapproche pour de grandes valeurs de n, et d'autres facteurs qui restent à déterminer.
