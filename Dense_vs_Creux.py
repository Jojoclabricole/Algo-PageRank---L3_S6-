"""
Implémentation et analyse de l'algorithme PageRank.
 
Fonctionnalités :
    Génération de graphes aléatoires (dense et creux)
    Calcul du PageRank
    Visualisation du graphe avec tailles de nœuds proportionnelles au PageRank
    Étude de la convergence en fonction du paramètre alpha
    Comparaison des performances dense vs creux
"""

from curses import ERR
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.creux import csc_matrix, csr_matrix


epsilon = 1e-16
alpha  = 0.85

# Génération de graphes aléatoires
def graph_alea(n, methode="dense"):
    """
    Génère une matrice d'adjacence aléatoire.
 
    methode="dense"  : matrice n*n binaire uniforme
                       Complexité : O(n^2)
 
    methode="creux" : chaque noeud a en moyenne log2(n) liens sortants.
                       Retourne une csc_matrix de forme (n, n).
                       Complexité : O(n*log n)
    """
    if methode == "dense":
        A= np.random.randint(2, size=(n, n))
 
    else
        d       = int(np.log2(n))   # degré moyen cible
        p       = d / n             # probabilité binomiale associée
        indptr  = [0]
        indices = []
        data    = []
 
        for j in range(n):
            k = np.random.binomial(n, p)   # nombre de liens sortants de j
            if k > 0:
                col = np.random.choice(n, size=k, replace=False)
                col.sort()
                indices.extend(col)
                data.extend([1] * k)
            indptr.append(len(indices))
        A = csc_matrix((data, indices, indptr), shape=(n, n))
 
    return A

# Converion de la matrice en graphe orienté NetworkX
def mat_to_graph(A, n, methode="dense"):
    """
    Convertit une matrice d'adjacence A en graphe orienté NetworkX.
    A[i,j] = 1 signifie qu'il y a un lien de la page j vers la page i.
 
    Pour methode="creux", A est convertie en dense avant parcours
    (usage réservé à la visualisation de petits graphes).
 
    Complexité : O(n^2)
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
 
    if methode == "creux":
 
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                G.add_edge(j, i)
 
    return G

# Vecteur initial du surfer
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

# Affichage du graphe
def affiche_graph(G) :
    """
    Affiche le graphe.

    Complexité : Complexité de la fonction draw : couteuse pour des grandes valeurs de n.
    """
    pos = nx.spring_layout(G) # Positionnement des nœuds pour une meilleure visualisation. Limite les chevauchements.
    nx.draw(G, pos, with_labels=True, node_size=800)
    #ligne suivante pour enregistrer le graphe et l'insérer dans le rapport.
    #plt.savefig("graphe.png")
    plt.show()
    print(G)
    return


# Matrice de transition P
def calcul_P(A, n, methode="dense"):
    """
    Calcule la matrice de transition P du graphe.
    P[i,j] = probabilité de passer de j vers i
 
    methode="dense"  : A est une matrice n*n
                       Complexité : O(n^2)
 
    methode="sparse" : A est une csr_matrix. (matrice creuse au format colonne)
                       Complexité : O(nnz(A)) = O(n*log n), avec nnz(A) le nombre d'entrées non nulles de A.
    """
    if methode == "dense":

        P = np.zeros((n, n))

        for j in range(n):

            #Calcul du degré sortant de j
            deg = np.sum(A[:, j])

            #remplissage de la colonne j de P
            if deg != 0:

                for i in range(n):

                    if A[i, j] == 1:
                        P[i, j] = 1.0 / deg
    else :
        rows = []
        cols = []
        data = []

        #parcours colonne par colonne
        for j in range(n):

            debut = A.indptr[j]
            fin = A.indptr[j+1]

            #degré de la colonne j :
            deg = fin - debut

            if deg > 0: 

                for i in A.indices[debut:fin] :
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0 / deg)

        P = csr_matrix((data, (rows, cols)), shape=(n, n))
    return P

# Algorithme de PageRank
def pagerank(A, n, x, methode="dense", alpha=0.85):
    """
    Calcule le vecteur PageRank par itération :

        x_{k+1} = alpha * P * x_k + (1-alpha)/n * 1

    jusqu'à convergence.

    Critère d'arrêt :
        max |x_{k+1} - x_k| < epsilon : toutes les composantes du vecteur de probabilité sont proches de leur valeur limite.

    Méthode "dense" : on utilise np.dot donc la complexité est en O(n^2 * k) avec k le nombre d'itérations nécessaires pour atteindre la convergence.
    Méthode "creux" : on utilise P.dot(x) donc la complexité est en O(nnz(P) * k) = O(n*log n * k).
    """

    teleport = (1 - alpha) / n
    erreurs = []

    P = calcul_P(A, n, methode)

    diff = 1

    while diff > epsilon :
        if methode == "dense":
            xnext = alpha * np.dot(P, x) + teleport
        else :
            xnext = alpha * P.dot(x) + teleport


        diff = np.max(np.abs(x - xnext))
        erreurs.append(diff)
        x = xnext
    
    return xnext, erreurs

# Visualisation du PageRank
def draw_pagerank(A, n, alpha, f=None, methode="dense"):
    """
    Affiche le graphe obtenu avec des tailles de nœuds proportionnelles au PageRank.
    """

    Ggraph = mat_to_graph(A, n, methode)

    Ggraph = calcul_G(A,n,alpha)
    x0 = surfer_alea(n)

    scores, _ = pagerank(A, n, x0, methode, alpha)

    sizes = 5000*scores

    nx.draw(Ggraph, with_labels=True, node_size=sizes)

    if f:
        plt.savefig(f)
    plt.show()
    
    return

# Étude de la convergence selon alpha
def convergence_alpha(A, n, x, alphas=None, methode="dense"):
    """
    Etudie la convergence de l'algorithme de PageRank en fonction de alpha.
    On remarque que plus alpha est proche de 1, plus la convergence est lente,
    cependant, plus alpha est proche de 1, plus le PageRank depend du graphe
    et moins du hasard.

    On cherche un bon candidat pour alpha, qui soit a la fois rapide a converger
    et qui reflète bien la structure du graphe.

    methode : "dense" ou "creux"
    """

    if alphas is None :
        alphas = [0.6,0.7,0.8,0.85,0.9,0.95]
    
        
    plt.figure()

    for a in alphas:

        _, erreurs = pagerank(A, n, x.copy(), methode, a)
        plt.semilogy(erreurs,label=f"alpha={a}")

    plt.xlabel("Iteration")
    plt.ylabel("Erreur (log)")
    plt.title("Convergence du PageRank selon alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_alpha.png")
    plt.show()

    return

# Comparaison des performances : dense vs creux
def comparaison(sizes, essais=100):
    """
    Mesure et compare les temps d'execution du PageRank dense et creux
    pour differentes tailles de graphe.

    Paramètres :
        sizes  : liste de valeurs de n a tester
        essais : nombre de repetitions pour moyenner les temps
    
    Les deux methodes partent du même graphe creux (pour plus de realisme), converti en dense, et du même vecteur de surfer initial.
    Affiche une courbe temps moyen en fonction de la taille du graphe.
    """
    temps_dense  = []
    temps_creux = []

    for n in sizes:
        print(f"Comparaison n={n}...")
        td, ts = 0.0, 0.0
        x = surfer_alea(n)

        for _ in range(essais):
            
            A_creux = graph_alea(n, "creux")
            A_dense = A_creux.toarray()

            # Dense
            start = time.time()

            pagerank(A_dense, n, x.copy(), "dense", alpha = alpha)
            td   += time.time() - start

            # Creux
            start = time.time()
            pagerank_creux(A_creux, n, x.copy(), "creux" alpha=alpha)
            ts   += time.time() - start

        temps_dense.append(td / essais)
        temps_creux.append(ts / essais)

    plt.figure()
    plt.plot(sizes, temps_dense, label="Dense")
    plt.plot(sizes, temps_creux, label="Creuse")
    plt.xlabel("Nombre de pages")
    plt.ylabel("Temps moyen (s)")
    plt.title("Comparaison PageRank dense vs creux")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparaison_dense_vs_creux.png")
    plt.show()
    return
"""
# Tests et programme principal

def main():
    # -----------------------------
    # Paramètres
    # -----------------------------
    n = 10
    alpha_values = [0.85]

    # -----------------------------
    # Génération des données
    # -----------------------------
    A = graph_alea(n)
    print("Matrice d'adjacence :")
    print(A)

    x0 = surfer_alea(n)

    # -----------------------------
    # Visualisation PageRank
    # -----------------------------
    for a in alpha_values:
        filename = f"pagerank_alpha_{int(a*100)}.png"
        draw_pagerank(A, n, a, filename)

    # -----------------------------
    # Étude de convergence
    # -----------------------------
    convergence_alpha(A, n, x0.copy())

    return
"""




# =============================================================================
# Programme principal - modifier cette section selon l'usage souhaite
# =============================================================================

def main():

    # ------------------------------------------------------------------
    # Paramètres
    # ------------------------------------------------------------------
    n      = 10
    alphas = [0.6, 0.7, 0.8, 0.9, 0.95]
    """
    # ------------------------------------------------------------------
    # Génération
    # ------------------------------------------------------------------
    A  = graph_creux(n)
    x0 = surfer_alea(n)

    print("Matrice d'adjacence :")
    print(A)
    
    # ------------------------------------------------------------------
    # Visualisation du graphe et du PageRank
    # ------------------------------------------------------------------
    affiche_graph(mat_to_graph(A, n))

    draw_pagerank(A, n, alpha,
                  filepath="pagerank_creux.png",
                  methode="creux")

    draw_pagerank(A, n, alpha,
                  filepath="pagerank_dense.png",
                  methode="dense")

    # ------------------------------------------------------------------
    # Etude de convergence selon alpha
    # ------------------------------------------------------------------
    convergence_alpha(A, n, x0.copy(), alphas=alphas, methode="creux")
    """
    # ------------------------------------------------------------------
    # Comparaison dense vs creux (decommenter pour lancer, couteux en temps)
    # ------------------------------------------------------------------
    comparaison(sizes=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], essais=1000)


if __name__ == "__main__":
    main()
