# Author : Ammari Abdelhafid
# Exercice 7 : Prise en compte de la structure du corpus
# M2 MIASHS : projet Network Analysis for Information Retrieval

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import os
import warnings
from collections import Counter, defaultdict
import pickle
warnings.filterwarnings('ignore')

# Pour les graphes
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from scipy import stats

# Pour le clustering et les embeddings de graphe
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Pour les visualisations interactives
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Pour Node2Vec (si disponible, sinon alternative de base)
try:
    from node2vec import Node2Vec
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False
    print("node2vec n'est pas installé. Une alternative basique sera utilisée.")


# Pour Louvain (si disponible, sinon spectral clustering)
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("python-louvain n'est pas installé. Spectral clustering sera utilisé à la place.")

# Pour la visualisation interactive des graphes
try:
    from IPython.display import display, HTML
    HAS_IPYSIGMA = True
except ImportError:
    HAS_IPYSIGMA = False
    print("ipysigma n'est pas installé. La visualisation du graphe sera limitée.")


# Configuration pour un meilleur affichage
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


#===========================================================================================
# 1. Chargement et préparation des données
#===========================================================================================

def load_processed_data(file_path):
    """
    Charge les données traitées lors des exercices précédents.

    Args:
        file_path (str): Chemin du fichier pickle contenant le DataFrame

    Returns:
        pandas.DataFrame: DataFrame contenant les articles
    """
    try:
        df = pd.read_pickle(file_path)
        print(f"Données chargées avec succès. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None

def identify_cluster_column(df):
    """
    Identifie la colonne contenant les étiquettes de cluster.

    Args:
        df (pandas.DataFrame): DataFrame contenant les articles

    Returns:
        str: Nom de la colonne de cluster
    """
    cluster_columns = [col for col in df.columns if 'cluster' in col.lower()]

    if not cluster_columns:
        print("Aucune colonne de cluster trouvée.")
        return None

    # S'il y a plusieurs colonnes de cluster, choisir celle qui semble la plus pertinente
    if len(cluster_columns) > 1:
        print(f"Plusieurs colonnes de cluster trouvées: {cluster_columns}")
        print(f"Utilisation de la première: {cluster_columns[0]}")

    return cluster_columns[0]

def identify_structural_columns(df):
    """
    Identifie les colonnes contenant des informations structurelles.

    Args:
        df (pandas.DataFrame): DataFrame contenant les articles

    Returns:
        dict: Dictionnaire des colonnes structurelles par type
    """
    structural_columns = {}

    # Recherche des colonnes d'auteurs
    author_columns = [col for col in df.columns if 'author' in col.lower()]
    if author_columns:
        structural_columns['authors'] = author_columns[0]

    # Recherche des colonnes de références
    ref_columns = [col for col in df.columns if 'reference' in col.lower()]
    if ref_columns:
        structural_columns['references'] = ref_columns[0]

    # Recherche des colonnes de publication
    venue_columns = [col for col in df.columns if any(term in col.lower() for term in ['venue', 'journal', 'conference'])]
    if venue_columns:
        structural_columns['venue'] = venue_columns[0]

    print(f"Colonnes structurelles identifiées: {structural_columns}")

    return structural_columns

#===========================================================================================
# 2. Construction du graphe
#===========================================================================================

class CorpusGraph:
    """Classe pour la construction et l'analyse d'un graphe de corpus scientifique."""

    def __init__(self, df, id_column='id'):
        """
        Initialise la classe avec le DataFrame d'articles.

        Args:
            df (pandas.DataFrame): DataFrame contenant les articles
            id_column (str): Nom de la colonne contenant les identifiants des articles
        """
        self.df = df
        self.id_column = id_column

        # Vérification de l'existence de la colonne ID
        if id_column not in df.columns:
            print(f"Colonne ID '{id_column}' non trouvée. Utilisation de l'index comme ID.")
            self.df['temp_id'] = df.index
            self.id_column = 'temp_id'

        # Création d'un mapping ID -> index
        self.id_to_index = {id_val: i for i, id_val in enumerate(df[self.id_column])}
        self.index_to_id = {i: id_val for i, id_val in enumerate(df[self.id_column])}

        # Initialisation du graphe
        self.graph = None
        self.adjacency_matrix = None

    def build_graph_from_authors(self, authors_column):
        """
        Construit un graphe basé sur les auteurs communs entre articles.

        Args:
            authors_column (str): Nom de la colonne contenant les auteurs des articles

        Returns:
            networkx.Graph: Graphe construit
        """
        print("Construction du graphe à partir des auteurs communs...")

        # Vérification de l'existence de la colonne d'auteurs
        if authors_column not in self.df.columns:
            print(f"Colonne d'auteurs '{authors_column}' non trouvée.")
            return None

        # Initialisation du graphe
        G = nx.Graph()

        # Ajout des nœuds (articles)
        for i, row in enumerate(tqdm(self.df.itertuples(), total=len(self.df), desc="Ajout des nœuds")):
            article_id = getattr(row, self.id_column)
            G.add_node(article_id, title=row.title, index=i)

        # Création d'un dictionnaire auteur -> liste d'articles
        author_to_articles = defaultdict(set)

        # Remplissage du dictionnaire
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des auteurs"):
            article_id = getattr(row, self.id_column)
            authors = getattr(row, authors_column)

            # Vérification du format des auteurs
            if isinstance(authors, list):
                for author in authors:
                    author_to_articles[author].add(article_id)
            elif isinstance(authors, str):
                # Si c'est une chaîne (peut-être séparée par des virgules)
                for author in authors.split(', '):
                    author_to_articles[author.strip()].add(article_id)

        # Création des arêtes pour les articles partageant des auteurs
        edges_to_add = []

        for articles in tqdm(author_to_articles.values(), total=len(author_to_articles), desc="Création des arêtes"):
            if len(articles) > 1:
                articles = list(articles)
                for i in range(len(articles)):
                    for j in range(i+1, len(articles)):
                        edges_to_add.append((articles[i], articles[j]))

        # Ajout des arêtes au graphe
        G.add_edges_from(edges_to_add)

        print(f"Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Stockage du graphe
        self.graph = G

        # Création de la matrice d'adjacence
        self.create_adjacency_matrix()

        return G

    def build_graph_from_references(self, references_column):
        """
        Construit un graphe basé sur les citations entre articles.

        Args:
            references_column (str): Nom de la colonne contenant les références des articles

        Returns:
            networkx.Graph: Graphe construit
        """
        print("Construction du graphe à partir des citations...")

        # Vérification de l'existence de la colonne de références
        if references_column not in self.df.columns:
            print(f"Colonne de références '{references_column}' non trouvée.")
            return None

        # Initialisation du graphe
        G = nx.DiGraph()  # Graphe dirigé pour les citations

        # Ajout des nœuds (articles)
        for i, row in enumerate(tqdm(self.df.itertuples(), total=len(self.df), desc="Ajout des nœuds")):
            article_id = getattr(row, self.id_column)
            G.add_node(article_id, title=row.title, index=i)

        # Ensemble des IDs d'articles dans le corpus
        article_ids = set(self.df[self.id_column])

        # Ajout des arêtes (citations)
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des références"):
            article_id = getattr(row, self.id_column)
            references = getattr(row, references_column)

            # Vérification du format des références
            if isinstance(references, list):
                for ref in references:
                    # Vérifier si la référence est un article du corpus
                    if ref in article_ids:
                        G.add_edge(article_id, ref)  # article_id cite ref
            elif isinstance(references, str):
                # Si c'est une chaîne (peut-être séparée par des virgules)
                for ref in references.split(', '):
                    ref = ref.strip()
                    if ref in article_ids:
                        G.add_edge(article_id, ref)

        print(f"Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Conversion en graphe non dirigé pour certaines analyses
        G_undirected = G.to_undirected()

        # Stockage du graphe
        self.graph = G_undirected

        # Création de la matrice d'adjacence
        self.create_adjacency_matrix()

        return G_undirected

    def build_graph_from_shared_references(self, references_column, threshold=1):
        """
        Construit un graphe basé sur les références bibliographiques partagées.

        Args:
            references_column (str): Nom de la colonne contenant les références des articles
            threshold (int): Nombre minimum de références partagées pour créer une arête

        Returns:
            networkx.Graph: Graphe construit
        """
        print("Construction du graphe à partir des références partagées...")

        # Vérification de l'existence de la colonne de références
        if references_column not in self.df.columns:
            print(f"Colonne de références '{references_column}' non trouvée.")
            return None

        # Initialisation du graphe
        G = nx.Graph()

        # Ajout des nœuds (articles)
        for i, row in enumerate(tqdm(self.df.itertuples(), total=len(self.df), desc="Ajout des nœuds")):
            article_id = getattr(row, self.id_column)
            G.add_node(article_id, title=row.title, index=i)

        # Création d'un dictionnaire article -> ensemble de références
        article_to_references = {}

        # Remplissage du dictionnaire
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des références"):
            article_id = getattr(row, self.id_column)
            references = getattr(row, references_column)

            # Vérification du format des références
            if isinstance(references, list):
                article_to_references[article_id] = set(references)
            elif isinstance(references, str):
                # Si c'est une chaîne
                article_to_references[article_id] = set(ref.strip() for ref in references.split(', '))
            else:
                # Si c'est un autre type ou None
                article_to_references[article_id] = set()

        # Création des arêtes pour les articles partageant des références
        edges_to_add = []

        article_ids = list(article_to_references.keys())

        for i in tqdm(range(len(article_ids)), desc="Création des arêtes"):
            for j in range(i+1, len(article_ids)):
                article1 = article_ids[i]
                article2 = article_ids[j]

                # Calcul du nombre de références partagées
                shared_refs = len(article_to_references[article1].intersection(article_to_references[article2]))

                # Ajout de l'arête si le nombre de références partagées dépasse le seuil
                if shared_refs >= threshold:
                    edges_to_add.append((article1, article2, {'weight': shared_refs}))

        # Ajout des arêtes au graphe
        G.add_edges_from(edges_to_add)

        print(f"Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Stockage du graphe
        self.graph = G

        # Création de la matrice d'adjacence
        self.create_adjacency_matrix()

        return G

    def build_graph_from_venue(self, venue_column):
        """
        Construit un graphe basé sur les articles publiés dans le même journal/conférence.

        Args:
            venue_column (str): Nom de la colonne contenant le lieu de publication des articles

        Returns:
            networkx.Graph: Graphe construit
        """
        print("Construction du graphe à partir des lieux de publication...")

        # Vérification de l'existence de la colonne de venue
        if venue_column not in self.df.columns:
            print(f"Colonne de venue '{venue_column}' non trouvée.")
            return None

        # Initialisation du graphe
        G = nx.Graph()

        # Ajout des nœuds (articles)
        for i, row in enumerate(tqdm(self.df.itertuples(), total=len(self.df), desc="Ajout des nœuds")):
            article_id = getattr(row, self.id_column)
            G.add_node(article_id, title=row.title, index=i)

        # Création d'un dictionnaire venue -> liste d'articles
        venue_to_articles = defaultdict(set)

        # Remplissage du dictionnaire
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des venues"):
            article_id = getattr(row, self.id_column)
            venue = getattr(row, venue_column)

            if pd.notna(venue) and venue:
                venue_to_articles[venue].add(article_id)

        # Création des arêtes pour les articles partageant la même venue
        edges_to_add = []

        for articles in tqdm(venue_to_articles.values(), total=len(venue_to_articles), desc="Création des arêtes"):
            if len(articles) > 1:
                articles = list(articles)
                for i in range(len(articles)):
                    for j in range(i+1, len(articles)):
                        edges_to_add.append((articles[i], articles[j]))

        # Ajout des arêtes au graphe
        G.add_edges_from(edges_to_add)

        print(f"Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Stockage du graphe
        self.graph = G

        # Création de la matrice d'adjacence
        self.create_adjacency_matrix()

        return G

    def build_combined_graph(self, structural_columns, weights=None):
        """
        Construit un graphe combinant plusieurs types de relations.

        Args:
            structural_columns (dict): Dictionnaire des colonnes structurelles par type
            weights (dict): Poids à attribuer à chaque type de relation

        Returns:
            networkx.Graph: Graphe combiné
        """
        print("Construction du graphe combiné...")

        # Poids par défaut si non spécifiés
        if weights is None:
            weights = {
                'authors': 1.0,
                'references': 0.5,
                'venue': 0.3
            }

        # Initialisation du graphe
        G = nx.Graph()

        # Ajout des nœuds (articles)
        for i, row in enumerate(tqdm(self.df.itertuples(), total=len(self.df), desc="Ajout des nœuds")):
            article_id = getattr(row, self.id_column)
            G.add_node(article_id, title=row.title, index=i)

        # Dictionnaire pour stocker les poids des arêtes
        edge_weights = defaultdict(float)

        # Traitement des auteurs
        if 'authors' in structural_columns:
            authors_column = structural_columns['authors']
            author_to_articles = defaultdict(set)

            # Remplissage du dictionnaire
            for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des auteurs"):
                article_id = getattr(row, self.id_column)
                authors = getattr(row, authors_column)

                if isinstance(authors, list):
                    for author in authors:
                        author_to_articles[author].add(article_id)
                elif isinstance(authors, str):
                    for author in authors.split(', '):
                        author_to_articles[author.strip()].add(article_id)

            # Ajout des poids pour les auteurs communs
            for articles in author_to_articles.values():
                if len(articles) > 1:
                    articles = list(articles)
                    for i in range(len(articles)):
                        for j in range(i+1, len(articles)):
                            edge = tuple(sorted([articles[i], articles[j]]))
                            edge_weights[edge] += weights['authors']

        # Traitement des références
        if 'references' in structural_columns:
            references_column = structural_columns['references']
            article_to_references = {}

            # Remplissage du dictionnaire
            for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des références"):
                article_id = getattr(row, self.id_column)
                references = getattr(row, references_column)

                if isinstance(references, list):
                    article_to_references[article_id] = set(references)
                elif isinstance(references, str):
                    article_to_references[article_id] = set(ref.strip() for ref in references.split(', '))
                else:
                    article_to_references[article_id] = set()

            # Ensemble des IDs d'articles dans le corpus
            article_ids = set(self.df[self.id_column])

            # Ajout des poids pour les citations directes
            for article_id, refs in article_to_references.items():
                for ref in refs:
                    if ref in article_ids:
                        edge = tuple(sorted([article_id, ref]))
                        edge_weights[edge] += weights['references']

            # Ajout des poids pour les références partagées
            article_ids = list(article_to_references.keys())

            for i in tqdm(range(len(article_ids)), desc="Traitement des références partagées"):
                for j in range(i+1, len(article_ids)):
                    article1 = article_ids[i]
                    article2 = article_ids[j]

                    # Calcul du nombre de références partagées
                    shared_refs = len(article_to_references[article1].intersection(article_to_references[article2]))

                    # Ajout du poids si des références sont partagées
                    if shared_refs > 0:
                        edge = tuple(sorted([article1, article2]))
                        edge_weights[edge] += weights['references'] * min(shared_refs, 5) / 5  # Plafonnement à 5 références

        # Traitement des venues
        if 'venue' in structural_columns:
            venue_column = structural_columns['venue']
            venue_to_articles = defaultdict(set)

            # Remplissage du dictionnaire
            for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Traitement des venues"):
                article_id = getattr(row, self.id_column)
                venue = getattr(row, venue_column)

                if pd.notna(venue) and venue:
                    venue_to_articles[venue].add(article_id)

            # Ajout des poids pour les venues communes
            for articles in venue_to_articles.values():
                if len(articles) > 1:
                    articles = list(articles)
                    for i in range(len(articles)):
                        for j in range(i+1, len(articles)):
                            edge = tuple(sorted([articles[i], articles[j]]))
                            edge_weights[edge] += weights['venue']

        # Ajout des arêtes avec leurs poids
        for edge, weight in tqdm(edge_weights.items(), desc="Ajout des arêtes"):
            G.add_edge(*edge, weight=weight)

        print(f"Graphe combiné construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Filtrage des arêtes faibles si le graphe est trop dense
        if G.number_of_edges() > 100000:
            weight_threshold = np.percentile([d['weight'] for u, v, d in G.edges(data=True)], 80)
            edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < weight_threshold]
            G.remove_edges_from(edges_to_remove)
            print(f"Graphe filtré: {G.number_of_edges()} arêtes restantes (seuil de poids: {weight_threshold:.4f})")

        # Stockage du graphe
        self.graph = G

        # Création de la matrice d'adjacence
        self.create_adjacency_matrix()

        return G

    def create_adjacency_matrix(self):
        """
        Crée la matrice d'adjacence du graphe.

        Returns:
            scipy.sparse.csr_matrix: Matrice d'adjacence
        """
        if self.graph is None:
            print("Le graphe n'a pas été construit.")
            return None

        print("Création de la matrice d'adjacence...")

        # Récupération des nœuds
        nodes = list(self.graph.nodes())
        n = len(nodes)

        # Création d'un mapping node -> index
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Initialisation de la matrice d'adjacence sparse
        adjacency = lil_matrix((n, n), dtype=float)

        # Remplissage de la matrice
        for u, v, data in tqdm(self.graph.edges(data=True), desc="Remplissage de la matrice"):
            i, j = node_to_idx[u], node_to_idx[v]
            weight = data.get('weight', 1.0)
            adjacency[i, j] = weight
            adjacency[j, i] = weight  # Graphe non dirigé

        # Conversion en format CSR pour les opérations efficaces
        self.adjacency_matrix = adjacency.tocsr()

        return self.adjacency_matrix

    def calculate_graph_statistics(self):
        """
        Calcule diverses statistiques sur le graphe.

        Returns:
            dict: Dictionnaire des statistiques calculées
        """
        if self.graph is None:
            print("Le graphe n'a pas été construit.")
            return None

        print("Calcul des statistiques du graphe...")

        G = self.graph

        stats = {}

        # Nombre de nœuds et d'arêtes
        stats['n_nodes'] = G.number_of_nodes()
        stats['n_edges'] = G.number_of_edges()

        # Densité du graphe
        stats['density'] = nx.density(G)

        # Degrés
        degrees = [d for n, d in G.degree()]
        stats['min_degree'] = min(degrees)
        stats['max_degree'] = max(degrees)
        stats['avg_degree'] = sum(degrees) / len(degrees)
        stats['median_degree'] = np.median(degrees)

        # Distribution des degrés (sous forme d'histogramme)
        degree_counts = Counter(degrees)
        stats['degree_distribution'] = sorted(degree_counts.items())

        # Composantes connexes
        components = list(nx.connected_components(G))
        stats['n_components'] = len(components)
        component_sizes = [len(c) for c in components]
        stats['largest_component_size'] = max(component_sizes)
        stats['largest_component_ratio'] = max(component_sizes) / stats['n_nodes']

        # Statistiques sur la composante géante
        if stats['largest_component_ratio'] > 0.1:  # Si la composante géante est significative
            giant = G.subgraph(list(components[component_sizes.index(max(component_sizes))]))

            # Diamètre (peut être coûteux pour les grands graphes)
            if giant.number_of_nodes() < 1000:
                stats['diameter'] = nx.diameter(giant)
            else:
                # Approximation du diamètre sur quelques nœuds
                sample_nodes = np.random.choice(list(giant.nodes()), min(100, giant.number_of_nodes()), replace=False)
                eccentricities = [max(nx.single_source_shortest_path_length(giant, node).values()) for node in sample_nodes]
                stats['approx_diameter'] = max(eccentricities)

            # Coefficient de clustering moyen
            if giant.number_of_nodes() < 10000:
                stats['avg_clustering'] = nx.average_clustering(giant)
            else:
                # Approximation sur un échantillon
                sample_nodes = np.random.choice(list(giant.nodes()), 1000, replace=False)
                clustering_coeffs = [nx.clustering(giant, node) for node in sample_nodes]
                stats['approx_avg_clustering'] = sum(clustering_coeffs) / len(clustering_coeffs)

            # Distance moyenne (peut être coûteuse pour les grands graphes)
            if giant.number_of_nodes() < 1000:
                stats['avg_shortest_path'] = nx.average_shortest_path_length(giant)
            else:
                # Approximation sur un échantillon
                sample_nodes = np.random.choice(list(giant.nodes()), 100, replace=False)
                path_lengths = []
                for u in sample_nodes:
                    for v in sample_nodes:
                        if u != v:
                            try:
                                path_lengths.append(nx.shortest_path_length(giant, u, v))
                            except nx.NetworkXNoPath:
                                pass
                if path_lengths:
                    stats['approx_avg_shortest_path'] = sum(path_lengths) / len(path_lengths)

        # Nodes with highest degree (hubs)
        degree_centrality = nx.degree_centrality(G)
        top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        stats['top_hubs'] = [(node, score) for node, score in top_hubs]

        print("Statistiques du graphe calculées.")

        return stats

    def filter_graph(self, min_degree=1, max_degree=None, largest_component_only=True):
        """
        Filtre le graphe selon certains critères.

        Args:
            min_degree (int): Degré minimum des nœuds à conserver
            max_degree (int): Degré maximum des nœuds à conserver
            largest_component_only (bool): Ne garder que la plus grande composante connexe

        Returns:
            networkx.Graph: Graphe filtré
        """
        if self.graph is None:
            print("Le graphe n'a pas été construit.")
            return None

        print("Filtrage du graphe...")

        G = self.graph.copy()
        original_size = G.number_of_nodes()

        # Filtrage par degré
        if min_degree > 1 or max_degree is not None:
            nodes_to_remove = []
            for node, degree in G.degree():
                if degree < min_degree:
                    nodes_to_remove.append(node)
                if max_degree is not None and degree > max_degree:
                    nodes_to_remove.append(node)

            G.remove_nodes_from(nodes_to_remove)
            print(f"Après filtrage par degré: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Extraction de la plus grande composante connexe
        if largest_component_only and G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            if len(components) > 1:
                largest_component = max(components, key=len)
                G = G.subgraph(largest_component).copy()
                print(f"Après extraction de la plus grande composante: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

        # Mise à jour du graphe
        self.graph = G

        # Recréation de la matrice d'adjacence
        self.create_adjacency_matrix()

        print(f"Graphe filtré: {G.number_of_nodes()} nœuds ({G.number_of_nodes()/original_size*100:.1f}% conservés)")

        return G

#===========================================================================================
# 3. Visualisation du graphe
#===========================================================================================

def visualize_graph_plotly(graph, node_colors=None, node_sizes=None, title="Visualisation du graphe", layout="spring", width=800, height=800):
    """
    Visualise un graphe avec Plotly.

    Args:
        graph (networkx.Graph): Graphe à visualiser
        node_colors (list): Couleurs des nœuds
        node_sizes (list): Tailles des nœuds
        title (str): Titre de la visualisation
        layout (str): Type de layout ('spring', 'kamada_kawai', 'spectral')
        width (int): Largeur de la figure
        height (int): Hauteur de la figure

    Returns:
        plotly.graph_objects.Figure: Figure Plotly
    """
    # Calcul du layout
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "spectral":
        pos = nx.spectral_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42)

    # Création des traces d'arêtes
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Création des traces de nœuds
    node_x = []
    node_y = []
    node_text = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Création du texte pour le hover
        node_attrs = graph.nodes[node]
        text = f"ID: {node}<br>"

        if 'title' in node_attrs:
            text += f"Title: {node_attrs['title']}<br>"

        node_text.append(text)

    # Configuration des couleurs des nœuds
    if node_colors is None:
        node_colors = 'blue'

    # Configuration des tailles des nœuds
    if node_sizes is None:
        node_sizes = 10

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Valeur',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )

    # Création de la figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height,
            autosize=False
        )
    )

    return fig

def visualize_graph_communities(graph, communities, title="Communautés dans le graphe", layout="spring", width=800, height=800):
    """
    Visualise les communautés d'un graphe avec Plotly.

    Args:
        graph (networkx.Graph): Graphe à visualiser
        communities (dict): Dictionnaire node -> community
        title (str): Titre de la visualisation
        layout (str): Type de layout ('spring', 'kamada_kawai', 'spectral')
        width (int): Largeur de la figure
        height (int): Hauteur de la figure

    Returns:
        plotly.graph_objects.Figure: Figure Plotly
    """
    # Calcul du layout
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "spectral":
        pos = nx.spectral_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42)

    # Création des traces d'arêtes
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Extraction des communautés uniques
    unique_communities = sorted(set(communities.values()))
    num_communities = len(unique_communities)

    # Création d'une palette de couleurs
    colorscale = px.colors.qualitative.Bold
    if num_communities > len(colorscale):
        colorscale = px.colors.qualitative.Dark24
    if num_communities > len(colorscale):
        colorscale = px.colors.qualitative.Alphabet

    # S'assurer qu'il y a assez de couleurs
    while len(colorscale) < num_communities:
        colorscale = colorscale * 2

    # Mapping communauté -> couleur
    community_colors = {comm: colorscale[i % len(colorscale)] for i, comm in enumerate(unique_communities)}

    # Création des traces de nœuds par communauté
    node_traces = []

    for community_id in unique_communities:
        # Filtrer les nœuds de cette communauté
        community_nodes = [node for node, comm in communities.items() if comm == community_id]

        node_x = []
        node_y = []
        node_text = []

        for node in community_nodes:
            if node in pos:  # S'assurer que le nœud est dans le layout
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                # Création du texte pour le hover
                node_attrs = graph.nodes[node]
                text = f"ID: {node}<br>Communauté: {community_id}<br>"

                if 'title' in node_attrs:
                    text += f"Title: {node_attrs['title']}<br>"

                node_text.append(text)

        # Création de la trace pour cette communauté
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            name=f"Communauté {community_id}",
            marker=dict(
                size=10,
                color=community_colors[community_id],
                line=dict(width=2)
            )
        )

        node_traces.append(node_trace)

    # Création de la figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height,
            autosize=False
        )
    )

    return fig

def plot_graph_statistics(graph_stats):
    """
    Visualise les statistiques du graphe.

    Args:
        graph_stats (dict): Dictionnaire des statistiques du graphe

    Returns:
        tuple: Tuple de figures Plotly
    """
    # Import explicite pour éviter le conflit de noms
    from scipy import stats as scipy_stats

    # Création des figures
    figures = []

    # Distribution des degrés
    if 'degree_distribution' in graph_stats:
        degree_dist = graph_stats['degree_distribution']

        degrees = [d for d, c in degree_dist]
        counts = [c for d, c in degree_dist]

        # Échelle log-log pour la loi de puissance
        fig_degree = go.Figure()

        fig_degree.add_trace(go.Scatter(
            x=degrees,
            y=counts,
            mode='markers',
            name='Distribution des degrés'
        ))

        # Ajout d'une ligne de tendance pour la loi de puissance
        if len(degrees) > 5 and len(counts) > 5:
            # Conversion en log
            log_degrees = np.log10(np.array(degrees) + 1)  # +1 pour éviter log(0)
            log_counts = np.log10(np.array(counts) + 1)  # +1 pour éviter log(0)

            # Régression linéaire en échelle log-log
            mask = np.isfinite(log_degrees) & np.isfinite(log_counts)
            if sum(mask) > 2:
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    log_degrees[mask], log_counts[mask]
                )

                # Génération des points pour la ligne
                x_range = np.linspace(min(log_degrees[mask]), max(log_degrees[mask]), 100)
                y_range = slope * x_range + intercept

                # Conversion depuis log
                x_orig = 10 ** x_range - 1
                y_orig = 10 ** y_range - 1

                fig_degree.add_trace(go.Scatter(
                    x=x_orig,
                    y=y_orig,
                    mode='lines',
                    name=f'Loi de puissance (α ≈ {-slope:.2f})'
                ))

        fig_degree.update_layout(
            title="Distribution des degrés",
            xaxis_title="Degré",
            yaxis_title="Nombre de nœuds",
            xaxis_type="log",
            yaxis_type="log",
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
        )

        figures.append(fig_degree)

    # Statistiques générales et table HTML
    # (reste de la fonction inchangé)

    general_stats = {
        'Nombre de nœuds': graph_stats.get('n_nodes', 0),
        'Nombre d\'arêtes': graph_stats.get('n_edges', 0),
        'Densité': graph_stats.get('density', 0),
        'Degré moyen': graph_stats.get('avg_degree', 0),
        'Degré médian': graph_stats.get('median_degree', 0),
        'Nombre de composantes': graph_stats.get('n_components', 0),
        'Taille de la plus grande composante': graph_stats.get('largest_component_size', 0),
        'Ratio de la plus grande composante': graph_stats.get('largest_component_ratio', 0)
    }

    if 'diameter' in graph_stats:
        general_stats['Diamètre'] = graph_stats['diameter']
    elif 'approx_diameter' in graph_stats:
        general_stats['Diamètre approximatif'] = graph_stats['approx_diameter']

    if 'avg_clustering' in graph_stats:
        general_stats['Coefficient de clustering moyen'] = graph_stats['avg_clustering']
    elif 'approx_avg_clustering' in graph_stats:
        general_stats['Coefficient de clustering moyen approximatif'] = graph_stats['approx_avg_clustering']

    if 'avg_shortest_path' in graph_stats:
        general_stats['Longueur moyenne du plus court chemin'] = graph_stats['avg_shortest_path']
    elif 'approx_avg_shortest_path' in graph_stats:
        general_stats['Longueur moyenne approximative du plus court chemin'] = graph_stats['approx_avg_shortest_path']

    # Création d'un tableau HTML
    table_html = "<table style='width:100%; border-collapse: collapse;'>"
    table_html += "<tr style='background-color: #f2f2f2;'><th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Statistique</th><th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Valeur</th></tr>"

    for stat, value in general_stats.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        table_html += f"<tr><td style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{stat}</td><td style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{formatted_value}</td></tr>"

    table_html += "</table>"

    return figures, table_html

#===========================================================================================
# 4. Clustering et représentations basées sur le graphe
#===========================================================================================

class GraphAnalyzer:

    """Classe pour l'analyse et le clustering de graphe."""

    def __init__(self, corpus_graph, df=None):
        """
        Initialise la classe avec un objet CorpusGraph.

        Args:
            corpus_graph (CorpusGraph): Objet contenant le graphe du corpus
            df (pandas.DataFrame): DataFrame contenant les articles (optionnel)
        """
        self.corpus_graph = corpus_graph
        self.graph = corpus_graph.graph
        self.adjacency_matrix = corpus_graph.adjacency_matrix
        self.df = df

        # Résultats
        self.spectral_clusters = None
        self.louvain_communities = None
        self.node_embeddings = None

    def create_node_embeddings_on_sample(self, sample_ratio=0.3, dimensions=64, **kwargs):
        """
        Crée des embeddings sur un échantillon du graphe puis interpole pour le reste.

        Args:
            sample_ratio (float): Proportion de nœuds à échantillonner (0-1)
            dimensions (int): Dimension des embeddings
            **kwargs: Autres paramètres pour DeepWalk

        Returns:
            dict: Dictionnaire node -> embedding
        """
        print(f"Création d'embeddings sur {sample_ratio*100:.1f}% du graphe...")

        # Échantillonnage des nœuds
        all_nodes = list(self.graph.nodes())
        n_sample = int(len(all_nodes) * sample_ratio)
        sampled_nodes = np.random.choice(all_nodes, n_sample, replace=False)

        # Création d'un sous-graphe
        subgraph = self.graph.subgraph(sampled_nodes)

        # Sauvegarde temporaire du graphe original
        original_graph = self.graph
        original_adj = self.adjacency_matrix

        # Remplacement temporaire par le sous-graphe
        self.graph = subgraph
        self.adjacency_matrix = None

        # Calcul des embeddings sur le sous-échantillon
        sample_embeddings = self.create_node_embeddings_deepwalk(
            dimensions=dimensions, **kwargs
        )

        # Restauration du graphe original
        self.graph = original_graph
        self.adjacency_matrix = original_adj

        # Extension des embeddings au graphe complet par propagation
        full_embeddings = {}

        # Pour les nœuds échantillonnés, utiliser directement leurs embeddings
        for node in sampled_nodes:
            full_embeddings[node] = sample_embeddings[node]

        # Pour les autres nœuds, calculer la moyenne des embeddings de leurs voisins échantillonnés
        for node in tqdm(set(all_nodes) - set(sampled_nodes), desc="Extension des embeddings"):
            neighbors = list(self.graph.neighbors(node))
            sampled_neighbors = [n for n in neighbors if n in sampled_nodes]

            if sampled_neighbors:
                # Moyenne des embeddings des voisins échantillonnés
                neighbor_embeds = [sample_embeddings[n] for n in sampled_neighbors]
                full_embeddings[node] = np.mean(neighbor_embeds, axis=0)
            else:
                # Si pas de voisins échantillonnés, utiliser un vecteur aléatoire
                full_embeddings[node] = np.random.randn(dimensions) / np.sqrt(dimensions)

        self.node_embeddings = full_embeddings
        return full_embeddings

    def run_spectral_clustering(self, n_clusters=8, random_state=42):
        """
        Applique le clustering spectral sur le graphe.

        Args:
            n_clusters (int): Nombre de clusters à créer
            random_state (int): Seed pour la reproductibilité

        Returns:
            dict: Dictionnaire node -> cluster
        """
        print(f"Application du clustering spectral avec {n_clusters} clusters...")

        if self.adjacency_matrix is None:
            print("La matrice d'adjacence n'a pas été créée.")
            return None

        # Récupération de la liste des nœuds
        nodes = list(self.graph.nodes())

        # Application du clustering spectral
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=random_state,
            n_init=10,
            assign_labels='discretize'
        )

        # Conversion des poids en similarités si nécessaire
        # (si les poids sont des distances ou des dissimilarités)
        # adjacency_sim = np.exp(-self.adjacency_matrix)  # Transformation exponentielle

        # Utilisation directe de la matrice d'adjacence comme matrice d'affinité
        cluster_labels = clustering.fit_predict(self.adjacency_matrix.toarray())

        # Création d'un dictionnaire node -> cluster
        node_clusters = {nodes[i]: int(label) for i, label in enumerate(cluster_labels)}

        # Stockage des résultats
        self.spectral_clusters = node_clusters

        print(f"Clustering spectral terminé. {len(set(node_clusters.values()))} clusters créés.")

        return node_clusters

    def run_louvain_clustering(self):
        """
        Applique l'algorithme de Louvain pour la détection de communautés.

        Returns:
            dict: Dictionnaire node -> communauté
        """
        if not HAS_LOUVAIN:
            print("La bibliothèque python-louvain n'est pas installée. Impossible d'exécuter l'algorithme de Louvain.")
            print("Pour l'installer: pip install python-louvain")
            return None

        print("Application de l'algorithme de Louvain...")

        # Application de l'algorithme de Louvain
        communities = community_louvain.best_partition(self.graph)

        # Stockage des résultats
        self.louvain_communities = communities

        print(f"Détection de communautés terminée. {len(set(communities.values()))} communautés trouvées.")

        return communities

    def create_node_embeddings_deepwalk(self, dimensions=64, walk_length=10, num_walks=80, workers=4):
        """
        Crée des embeddings de nœuds avec DeepWalk (alternative basique).

        Args:
            dimensions (int): Dimension des embeddings
            walk_length (int): Longueur des marches aléatoires
            num_walks (int): Nombre de marches aléatoires par nœud
            workers (int): Nombre de threads parallèles

        Returns:
            dict: Dictionnaire node -> embedding
        """
        if HAS_NODE2VEC:
            print("Création des embeddings de nœuds avec Node2Vec...")

            # Configuration de Node2Vec
            node2vec = Node2Vec(
                self.graph,
                dimensions=dimensions,
                walk_length=walk_length,
                num_walks=num_walks,
                workers=workers
            )

            # Entraînement du modèle
            model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=4
            )

            # Récupération des embeddings
            node_embeddings = {}
            for node in self.graph.nodes():
                try:
                    embedding = model.wv[str(node)]  # Conversion en chaîne si nécessaire
                    node_embeddings[node] = embedding
                except KeyError:
                    # Si le nœud n'a pas d'embedding (rare)
                    node_embeddings[node] = np.zeros(dimensions)

            print(f"Embeddings créés pour {len(node_embeddings)} nœuds.")

            # Stockage des résultats
            self.node_embeddings = node_embeddings

            return node_embeddings
        else:
            print("Node2Vec n'est pas installé. Utilisation d'une alternative basique...")

            # Implémentation basique inspirée de DeepWalk
            # 1. Génération de marches aléatoires
            walks = []

            for _ in tqdm(range(num_walks), desc="Génération des marches"):
                nodes = list(self.graph.nodes())
                np.random.shuffle(nodes)

                for node in nodes:
                    # Marche aléatoire à partir de ce nœud
                    walk = [node]

                    for _ in range(walk_length - 1):
                        current = walk[-1]
                        neighbors = list(self.graph.neighbors(current))

                        if not neighbors:
                            break

                        walk.append(np.random.choice(neighbors))

                    walks.append([str(n) for n in walk])  # Conversion en chaîne

            # 2. Entraînement d'un modèle Word2Vec sur ces marches
            from gensim.models import Word2Vec

            model = Word2Vec(
                walks,
                vector_size=dimensions,
                window=10,
                min_count=1,
                sg=1,  # Skip-gram
                workers=workers,
                epochs=5
            )

            # 3. Récupération des embeddings
            node_embeddings = {}
            for node in self.graph.nodes():
                try:
                    embedding = model.wv[str(node)]
                    node_embeddings[node] = embedding
                except KeyError:
                    node_embeddings[node] = np.zeros(dimensions)

            print(f"Embeddings créés pour {len(node_embeddings)} nœuds.")

            # Stockage des résultats
            self.node_embeddings = node_embeddings

            return node_embeddings

    def project_embeddings(self, method='tsne', perplexity=30, n_components=2, random_state=42):
        """
        Projette les embeddings de nœuds en 2D pour la visualisation.

        Args:
            method (str): Méthode de projection ('tsne' ou 'pca')
            perplexity (int): Paramètre de perplexité pour t-SNE
            n_components (int): Nombre de composantes
            random_state (int): Seed pour la reproductibilité

        Returns:
            dict: Dictionnaire node -> coordonnées 2D
        """
        if self.node_embeddings is None:
            print("Les embeddings de nœuds n'ont pas été créés.")
            return None

        print(f"Projection des embeddings avec {method}...")

        # Extraction des embeddings et des nœuds correspondants
        nodes = list(self.node_embeddings.keys())
        embeddings = np.array([self.node_embeddings[node] for node in nodes])

        # Application de la méthode de projection
        if method == 'tsne':
            projection = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            ).fit_transform(embeddings)
        elif method == 'pca':
            projection = PCA(
                n_components=n_components,
                random_state=random_state
            ).fit_transform(embeddings)
        else:
            print(f"Méthode '{method}' non reconnue. Utilisation de t-SNE.")
            projection = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            ).fit_transform(embeddings)

        # Création d'un dictionnaire node -> coordonnées
        node_coords = {nodes[i]: projection[i] for i in range(len(nodes))}

        return node_coords

    def visualize_embeddings(self, node_coords, node_colors=None, node_clusters=None, title="Projection des embeddings de nœuds"):
        """
        Visualise les embeddings de nœuds projetés.

        Args:
            node_coords (dict): Dictionnaire node -> coordonnées 2D
            node_colors (dict): Dictionnaire node -> couleur
            node_clusters (dict): Dictionnaire node -> cluster
            title (str): Titre de la visualisation

        Returns:
            plotly.graph_objects.Figure: Figure Plotly
        """
        # Extraction des nœuds et des coordonnées
        nodes = list(node_coords.keys())
        coords = np.array([node_coords[node] for node in nodes])

        # Configuration des couleurs
        if node_clusters is not None:
            # Utilisation des clusters comme couleurs
            clusters = [node_clusters.get(node, -1) for node in nodes]
            unique_clusters = sorted(set(clusters))

            # Création d'une palette de couleurs
            colorscale = px.colors.qualitative.Bold
            if len(unique_clusters) > len(colorscale):
                colorscale = px.colors.qualitative.Dark24
            if len(unique_clusters) > len(colorscale):
                colorscale = px.colors.qualitative.Alphabet

            # S'assurer qu'il y a assez de couleurs
            while len(colorscale) < len(unique_clusters):
                colorscale = colorscale * 2

            # Création de la figure avec des traces par cluster
            fig = go.Figure()

            for cluster_id in unique_clusters:
                # Filtrer les nœuds de ce cluster
                indices = [i for i, c in enumerate(clusters) if c == cluster_id]

                if not indices:
                    continue

                cluster_nodes = [nodes[i] for i in indices]
                cluster_coords = coords[indices]

                # Textes pour le hover
                texts = []
                for node in cluster_nodes:
                    text = f"ID: {node}<br>Cluster: {cluster_id}"

                    if self.df is not None:
                        # Récupération des métadonnées si disponibles
                        try:
                            article = self.df[self.df[self.corpus_graph.id_column] == node].iloc[0]
                            text += f"<br>Title: {article['title']}"
                        except (IndexError, KeyError):
                            pass

                    texts.append(text)

                # Ajout de la trace pour ce cluster
                fig.add_trace(go.Scatter(
                    x=cluster_coords[:, 0],
                    y=cluster_coords[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colorscale[unique_clusters.index(cluster_id) % len(colorscale)]
                    ),
                    text=texts,
                    hoverinfo='text',
                    name=f"Cluster {cluster_id}"
                ))

            fig.update_layout(
                title=title,
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                showlegend=True,
                width=800,
                height=800
            )

            return fig

        else:
            # Utilisation d'une seule couleur ou de couleurs fournies
            if node_colors is None:
                colors = 'blue'
            else:
                colors = [node_colors.get(node, 'blue') for node in nodes]

            # Textes pour le hover
            texts = []
            for node in nodes:
                text = f"ID: {node}"

                if self.df is not None:
                    # Récupération des métadonnées si disponibles
                    try:
                        article = self.df[self.df[self.corpus_graph.id_column] == node].iloc[0]
                        text += f"<br>Title: {article['title']}"
                    except (IndexError, KeyError):
                        pass

                texts.append(text)

            # Création de la figure
            fig = go.Figure(data=go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True if isinstance(colors, list) else False
                ),
                text=texts,
                hoverinfo='text'
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                width=800,
                height=800
            )

            return fig

    def compare_clusterings(self, text_cluster_column=None):
        """
        Compare les différents clustering réalisés.

        Args:
            text_cluster_column (str): Nom de la colonne contenant les clusters basés sur le texte

        Returns:
            pandas.DataFrame: DataFrame contenant les scores de comparaison
        """
        print("Comparaison des différents clusterings...")

        # Liste des clusterings à comparer
        clusterings = {}

        if self.spectral_clusters is not None:
            clusterings['spectral'] = self.spectral_clusters

        if self.louvain_communities is not None:
            clusterings['louvain'] = self.louvain_communities

        # Ajout des clusters basés sur le texte si disponibles
        if text_cluster_column is not None and self.df is not None:
            if text_cluster_column in self.df.columns:
                # Création d'un dictionnaire node -> cluster textuel
                text_clusters = {}

                for row in self.df.itertuples():
                    node_id = getattr(row, self.corpus_graph.id_column)
                    if node_id in self.graph.nodes():
                        text_clusters[node_id] = getattr(row, text_cluster_column)

                clusterings['text'] = text_clusters

        # Si moins de 2 clusterings, impossible de comparer
        if len(clusterings) < 2:
            print("Pas assez de clusterings disponibles pour la comparaison.")
            return None

        # Création d'une matrice de comparaison
        comparison_scores = []

        for name1, clusters1 in clusterings.items():
            for name2, clusters2 in clusterings.items():
                if name1 != name2:
                    # Récupération des nœuds communs
                    common_nodes = set(clusters1.keys()).intersection(set(clusters2.keys()))

                    if not common_nodes:
                        continue

                    # Création des listes de labels
                    labels1 = [clusters1[node] for node in common_nodes]
                    labels2 = [clusters2[node] for node in common_nodes]

                    # Calcul des scores
                    ari = adjusted_rand_score(labels1, labels2)
                    nmi = normalized_mutual_info_score(labels1, labels2)

                    comparison_scores.append({
                        'clustering1': name1,
                        'clustering2': name2,
                        'nodes_count': len(common_nodes),
                        'adjusted_rand_index': ari,
                        'normalized_mutual_info': nmi
                    })

        # Création du DataFrame
        df_comparison = pd.DataFrame(comparison_scores)

        print("Comparaison terminée.")

        return df_comparison

    def add_clusters_to_dataframe(self, spectral_column='cluster_spectral', louvain_column='cluster_louvain'):
        """
        Ajoute les clusters calculés au DataFrame.

        Args:
            spectral_column (str): Nom de la colonne pour les clusters spectraux
            louvain_column (str): Nom de la colonne pour les communautés Louvain

        Returns:
            pandas.DataFrame: DataFrame avec les colonnes de clusters ajoutées
        """
        if self.df is None:
            print("Aucun DataFrame fourni.")
            return None

        # Copie du DataFrame
        df_with_clusters = self.df.copy()

        # Ajout des clusters spectraux
        if self.spectral_clusters is not None:
            df_with_clusters[spectral_column] = df_with_clusters[self.corpus_graph.id_column].map(
                lambda x: self.spectral_clusters.get(x, -1)
            )

        # Ajout des communautés Louvain
        if self.louvain_communities is not None:
            df_with_clusters[louvain_column] = df_with_clusters[self.corpus_graph.id_column].map(
                lambda x: self.louvain_communities.get(x, -1)
            )

        return df_with_clusters

