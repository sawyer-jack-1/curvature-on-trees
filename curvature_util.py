import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import squareform, pdist
import cvxpy

#####################################################
#####################################################
#####################################################
#####################################################

def distance_matrix(G: nx.Graph):

    nodes = list(nx.nodes(G))
    n = len(nodes)
    dist_mat = np.zeros((n, n))

    dist_dict = dict(nx.all_pairs_shortest_path_length(G))

    for count_i, n_i in enumerate(nodes):
        for count_j, n_j in enumerate(nodes):

            if count_j >= count_i:
                
                dist_mat[count_i, count_j] = dist_dict[n_i][n_j]
                dist_mat[count_j, count_i] = dist_dict[n_i][n_j]

    return dist_mat

def resistance_matrix(G: nx.Graph):

    nodes = list(nx.nodes(G))
    n = len(nodes)

    L = nx.laplacian_matrix(G).todense()
    Lambda, U = np.linalg.eigh(L)

    Lambda_half = np.diag([l**(-0.5) if l > 1e-6 else 0 for l in Lambda])
    embed = Lambda_half @ U.T

    R = squareform(pdist(embed.T)) **2

    return R


def solve_w1(G: nx.Graph, mu: np.ndarray, nu: np.ndarray):

    n = G.number_of_nodes()
    m = G.number_of_edges()

    w = np.ones((m, 1))

    w_dict = nx.get_edge_attributes(G, name='weight')
    if len(w_dict) > 0:
        for count, e in enumerate(nx.edges(G)):
            w[count, 0] = w_dict[e]

    B = nx.incidence_matrix(G, oriented=True).todense()
    J = cvxpy.Variable((m, 1))
    eps = 1e-8

    P = cvxpy.Minimize(cvxpy.norm( cvxpy.multiply(J, w), 1) + eps*cvxpy.power(cvxpy.norm(J, 2), 2))
    problem = cvxpy.Problem(P, [(B @ J) == mu - nu])

    problem.solve()

    return float(problem.value), J.value

def one_step_transition_cols(G: nx.Graph, alpha: float):

    n = G.number_of_nodes()
    m = G.number_of_edges()

    w_dict = nx.get_edge_attributes(G, name='weight')
    use_weights = (len(w_dict) > 0)

    d_vec = nx.adjacency_matrix(G).todense().sum(axis=0)

    one_step_transition = np.zeros((n, n))

    for count, i in enumerate(G.nodes()):

        nbrs = list(nx.all_neighbors(G, i))
        mu_i_alpha = one_step_transition[:, count]
        mu_i_alpha[count] = alpha

        for nbr in nbrs:

            if use_weights:
                try:
                    w_ij = w_dict[(i, nbr)]
                except KeyError:
                    w_ij = w_dict[(nbr, i)]
            else:
                w_ij = 1.0

            j_idx = list(nx.nodes(G)).index(nbr)

            mu_i_alpha[j_idx] = ((1-alpha) * w_ij) / d_vec[count]

    return one_step_transition

def formula_w1_onestep_trees(G: nx.Graph, node_i, node_j, alpha: float):

    n = G.number_of_nodes()
    m = G.number_of_edges()
    mus = one_step_transition_cols(G, alpha=alpha)

    i, j = list(G.nodes()).index(node_i), list(G.nodes()).index(node_j)
    mu, nu = mus[:, i][:, None], mus[:, j][:, None]

    w = np.ones((m, 1))

    w_dict = nx.get_edge_attributes(G, name='weight')
    if len(w_dict) > 0:
        for count, e in enumerate(nx.edges(G)):
            w[count, 0] = w_dict[e]

    geod = nx.shortest_path(G, node_i, node_j)
    geod_e = [ ( (geod[i], geod[i+1]) if (geod[i], geod[i+1]) in list(G.edges()) else (geod[i+1], geod[i])) for i in range(len(geod)-1)]
    geod_e_idx =  [list(G.edges()).index(geo_e) for geo_e in geod_e]
    sgn_es = np.array([-1.0 if e_idx in geod_e_idx else 1.0 for e_idx in range(len(w))])

    geod_e_indic = np.array([1.0 if e_idx in geod_e_idx else 0.0 for e_idx in range(len(w))])
    weighted_dist = (w.flatten() * geod_e_indic).sum()

    # print(f"Geodesic: {geod, geod_e, geod_e_idx}")
    # print(f"Geodesic indicator: {geod_e_indic}")
    # print(f"weights: {w}")
    # print(f"weights times indicator: {w.flatten()*geod_e_indic}")
    # print(f"Weighted dist: {weighted_dist:>3.1f}")

    terms = []

    for node in [node_i, node_j]:

        node_idx = list(G.nodes()).index(node)

        nbrs = list(G.neighbors(node))

        edges_node = [((node, nbr) if (node, nbr) in list(G.edges()) else (nbr, node)) for nbr in nbrs]
        edges_idx =  [list(G.edges()).index(nbr_e) for nbr_e in edges_node]
        edges_indic = np.array([1.0 if e_idx in edges_idx else 0.0 for e_idx in range(len(w))])

        term = (edges_indic * sgn_es *(w.flatten() ** 2)).sum()

        term /= nx.adjacency_matrix(G).todense().sum(axis=0)[node_idx]

        terms.append(term)

    w1 = weighted_dist + (1-alpha)*(np.sum(terms))
    
    return w1

def alternate_formula_w1_onestep_trees(G: nx.Graph, node_i, node_j, alpha: float):

    # assert ((node_i, node_j) in G.edges()) and (alpha < 0.5)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    mus = one_step_transition_cols(G, alpha=alpha)

    i, j = list(G.nodes()).index(node_i), list(G.nodes()).index(node_j)
    mu, nu = mus[:, i][:, None], mus[:, j][:, None]

    w = np.ones((m, 1))

    w_dict = nx.get_edge_attributes(G, name='weight')
    if len(w_dict) > 0:
        for count, e in enumerate(nx.edges(G)):
            w[count, 0] = w_dict[e]

    ij_idx = (list(G.edges()).index((node_i, node_j)) if (node_i, node_j) in list(G.edges()) else list(G.edges()).index((node_j, node_i)))
    w_ij = w[ij_idx]

    d_i, d_j = nx.adjacency_matrix(G).todense().sum(axis=0)[i], nx.adjacency_matrix(G).todense().sum(axis=0)[j]

    leading_term = w_ij * np.abs(1 - (1-alpha) * w_ij * (1 / d_i + 1 / d_j))

    terms = []

    for node in [node_i, node_j]:

        node_idx = list(G.nodes()).index(node)

        nbrs = list(G.neighbors(node))
        nbrs.remove((node_i if node == node_j else node_j))

        edges_node = [((node, nbr) if (node, nbr) in list(G.edges()) else (nbr, node)) for nbr in nbrs]
        edges_idx =  [list(G.edges()).index(nbr_e) for nbr_e in edges_node]
        edges_indic = np.array([1.0 if e_idx in edges_idx else 0.0 for e_idx in range(len(w))])

        term = (edges_indic * (w.flatten() ** 2)).sum()

        term /= nx.adjacency_matrix(G).todense().sum(axis=0)[node_idx]

        terms.append(term)

    print(terms, leading_term)

    w1 = leading_term + (1-alpha)*(np.sum(terms))
    
    return w1

#####################################################
#####################################################
#####################################################
#####################################################

def ste_curvature(G: nx.Graph):

    D = distance_matrix(G)

    o = nx.number_of_nodes(G) * np.ones( (nx.number_of_nodes(G), 1))

    return (np.linalg.pinv(D) @ o).flatten()

def dev_ott_ste_curvature(G: nx.Graph):

    R = resistance_matrix(G)
    o = nx.number_of_nodes(G) * np.ones( (nx.number_of_nodes(G), 1))

    return (np.linalg.pinv(R) @ o).flatten()

def dev_lam_curvature(G: nx.Graph):

    A = nx.adjacency_matrix(G).todense()
    R = resistance_matrix(G)

    return 1 - 0.5 * (A * R).sum(axis=0)

def oll_ric_curvature(G: nx.Graph, alpha: float):

    mu_alphas = one_step_transition_cols(G, alpha=alpha)
    kappas = {}
    w_dict = nx.get_edge_attributes(G, name='weight')

    for e_count, (node_i, node_j) in enumerate(G.edges()):

        node_i_idx = list(G.nodes()).index(node_i)
        node_j_idx = list(G.nodes()).index(node_j)

        mu_i = mu_alphas[:, node_i_idx][:, None]
        mu_j = mu_alphas[:, node_j_idx][:, None]

        w_1_ij = solve_w1(G, mu_i, mu_j)[0]

        if len(w_dict) > 0:
            w_ij = w_dict[(node_i, node_j)]
        else:
            w_ij = 1
        kappas[(node_i, node_j)] = 1 - w_1_ij / w_ij

    return kappas

def lin_lu_yau_curvature(G: nx.Graph):

    alpha = 1-1e-1

    mu_alphas = one_step_transition_cols(G, alpha=alpha)
    kappas = {}
    w_dict = nx.get_edge_attributes(G, name='weight')

    orc_alpha = oll_ric_curvature(G, alpha=alpha)

    for e_count, e in enumerate(G.edges()):

        kappas[e] = orc_alpha[e] / (1-alpha)

    return kappas

def prism_graph(k):

    G = nx.Graph()

    for i in range(k):

        G.add_edge(i, np.mod(i+1, k))
        G.add_edge(i+k, np.mod(i+1, k)+k)
        G.add_edge(i, i+k)

    return G

def orc_heatkernel_curvature(G: nx.Graph, t: float):

    L = nx.laplacian_matrix(G).todense()
    U, S, Vh = np.linalg.svd(L)

    e_tL = (U @ np.diag(np.exp(-t * S))) @ Vh

    kappas = {}

    w_dict = nx.get_edge_attributes(G, name='weight')

    for e_count, (node_i, node_j) in enumerate(G.edges()):

        node_i_idx = list(G.nodes()).index(node_i)
        node_j_idx = list(G.nodes()).index(node_j)

        mu_i = e_tL[:, node_i_idx][:, None]
        mu_j = e_tL[:, node_j_idx][:, None]

        w_1_ij = solve_w1(G, mu_i, mu_j)[0]

        if len(w_dict) > 0:
            w_ij = w_dict[(node_i, node_j)]
        else:
            w_ij = 1

        kappas[(node_i, node_j)] = 1 - w_1_ij / w_ij

    return kappas
