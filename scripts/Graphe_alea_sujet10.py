import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

epsilon = 1e-16



def graph_alea(n) :
    """
    Génère une matrice d'adjacence aléatoire.

    Complexité :
    O(n^2) pour générer la matrice.
    """

    A = np.random.randint(2, size = (n,n))

    return A

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
            if A[i,j] == 1 :
                G.add_edge(j,i) 
                # On ajoute une arête de j vers i car A[i,j] = 1 donc il y a un lien de la page j vers la page i.

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
                
                if A[i,j] == 1:
                    P[i,j] = 1 / deg

    return P


def calcul_G(A,n,alpha) :
    """
    Construit la matrice de Google :

        G = αP + (1-α)/n * U

    où (1-α)/n * U est la matrice de téléportation.

    Complexité :
    O(n^2)
    """

    P = calcul_P(A,n)
    U = np.ones([n,n])

    G = alpha*P + 1/n * (1 - alpha) * U

    return G




def pagerank(A,n,x,alpha) :
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
    
    G = calcul_G(A,n,alpha)

    xnext = np.dot(G,x)
    erreurs = []

    diff = np.max(np.abs(x - xnext))
    erreurs.append(diff)

    while diff > epsilon :
    
      x = xnext
      xnext = np.dot(G,x)

      diff = np.max(np.abs(x - xnext))
      erreurs.append(diff)
    
    return xnext, erreurs

def draw_pagerank(A,n,x,alpha,f):
    """
    Utilise l'algorithme de PageRank et le graphe obtenu avec des tailles de nœuds proportionnelles au PageRank.
    """

    Ggraph = mat_to_graph(A,n)

    scores, _ = pagerank(A,n,x,alpha)

    sizes = 5000*scores

    nx.draw(Ggraph,
            with_labels=True,
            node_size=sizes)

    plt.savefig(f)
    plt.show()
    
    return

def convergence_alpha(A,n,x):
    """
    Étudie la convergence de l'algorithme de PageRank en fonction de alpha.
    On remarque que plus alpha est proche de 1, plus la convergence est lente, cependant, plus alpha est proche de 1, plus le PageRank dépend du graphe et moins du hasard.

    On cherche un bon candidat pour alpha, qui soit à la fois rapide à converger et qui reflète bien la structure du graphe.
    """

    alphas = [0.6,0.7,0.8,0.9,0.95]
    x_copy = x.copy()

    plt.figure()

    for a in alphas:

        G = calcul_G(A,n,a)

        _, erreurs = pagerank(G,n,x_copy)
        plt.semilogy(erreurs,label=f"alpha={a}")

    plt.xlabel("Iteration")
    plt.ylabel("Erreur (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_alpha.png")
    plt.show()

    return
    

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
        draw_pagerank(A, n, x0, a, filename)

    # -----------------------------
    # Étude de convergence
    # -----------------------------
    convergence_alpha(A, n, x0.copy())

    return

if __name__ == "__main__":
    main()
