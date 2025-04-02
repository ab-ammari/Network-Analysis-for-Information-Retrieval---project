# Exercice 8 : Classification supervisée
# M2 MIASHS : projet Network Analysis for Information Retrieval

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import re
import warnings
from collections import Counter
import pickle
warnings.filterwarnings('ignore')

# Pour le traitement du texte
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Pour les graphes
import networkx as nx
from scipy.sparse import csr_matrix, hstack, vstack

# Pour les modèles de classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

# Pour les visualisations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

# Pour NLTK et gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from collections import defaultdict


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

def prepare_text_column(df, text_column='combined_text'):
    """
    Prépare une colonne de texte combiné (titre + résumé) si nécessaire.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les articles
        text_column (str): Nom de la colonne à créer
        
    Returns:
        pandas.DataFrame: DataFrame avec la colonne de texte combiné
    """
    df_copy = df.copy()
    
    # Vérifier si la colonne existe déjà
    if text_column not in df_copy.columns:
        print(f"Création de la colonne '{text_column}'...")
        # Combiner le titre et le résumé
        df_copy[text_column] = df_copy.apply(
            lambda row: f"{row['title']} {row.get('abstract', '') if pd.notna(row.get('abstract', '')) else ''}",
            axis=1
        )
    
    return df_copy

def explore_class_distribution(df, class_column='class'):
    """
    Explore la distribution des classes et crée une visualisation.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les articles
        class_column (str): Nom de la colonne contenant les étiquettes de classe
        
    Returns:
        plotly.graph_objects.Figure: Figure de la distribution des classes
    """
    if class_column not in df.columns:
        print(f"Colonne de classe '{class_column}' non trouvée.")
        return None
    
    # Mapping des classes
    class_mapping = {
        1: "Artificial Intelligence",
        2: "Data Science",
        3: "Interface",
        4: "Computer Vision",
        5: "Network",
        6: "Theoretical CS",
        7: "Specific Applications",
        8: "Other"
    }
    
    # Distribution des classes
    class_counts = df[class_column].value_counts().sort_index()
    
    # Création d'un DataFrame pour la visualisation
    df_viz = pd.DataFrame({
        'class_id': class_counts.index,
        'count': class_counts.values,
        'class_name': [class_mapping.get(c, f"Class {c}") for c in class_counts.index]
    })
    
    # Création de la figure
    fig = px.bar(
        df_viz,
        x='class_id',
        y='count',
        color='class_name',
        labels={'class_id': 'Classe', 'count': 'Nombre d\'articles', 'class_name': 'Domaine'},
        title='Distribution des classes dans le corpus',
        text='count'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500, width=900)
    
    # Afficher les statistiques
    total = df_viz['count'].sum()
    print(f"Nombre total d'articles classifiés: {total}")
    
    for i, row in df_viz.iterrows():
        print(f"Classe {row['class_id']} ({row['class_name']}): {row['count']} articles ({row['count']/total*100:.1f}%)")
    
    return fig

#===========================================================================================
# 2. Extraction de features textuelles
#===========================================================================================

class TextFeatureExtractor:
    """Classe pour l'extraction de features textuelles."""
    
    def __init__(self, df, text_column='combined_text', class_column='class'):
        """
        Initialise la classe avec le DataFrame d'articles.
        
        Args:
            df (pandas.DataFrame): DataFrame contenant les articles
            text_column (str): Nom de la colonne contenant le texte
            class_column (str): Nom de la colonne contenant les étiquettes de classe
        """
        self.df = df
        self.text_column = text_column
        self.class_column = class_column
        
        # Vérification des colonnes
        if text_column not in df.columns:
            raise ValueError(f"Colonne de texte '{text_column}' non trouvée dans le DataFrame.")
        
        if class_column not in df.columns:
            raise ValueError(f"Colonne de classe '{class_column}' non trouvée dans le DataFrame.")
        
        # Prétraitement
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Ajout de stop words spécifiques au domaine scientifique
        scientific_stop_words = [
            'doi', 'fig', 'figure', 'et', 'al', 'paper', 'study', 'research',
            'method', 'results', 'analysis', 'data', 'proposed', 'approach',
            'using', 'based', 'used', 'show', 'shown', 'table', 'section'
        ]
        self.stop_words.update(scientific_stop_words)
        
        # Attributs pour stocker les résultats
        self.vectorizer = None
        self.features = None
        self.feature_names = None
    
    def preprocess_text(self, text):
        """
        Prétraite un texte (minuscules, retrait ponctuation, etc.)
        
        Args:
            text (str): Texte à prétraiter
            
        Returns:
            str: Texte prétraité
        """
        if pd.isna(text):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des chiffres et de la ponctuation
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenise et applique le stemming au texte.
        
        Args:
            text (str): Texte à tokeniser
            
        Returns:
            list: Liste de stems
        """
        words = word_tokenize(text)
        return [self.stemmer.stem(word) for word in words if word.lower() not in self.stop_words]
    
    def extract_tfidf_features(self, min_df=5, max_df=0.95, max_features=None, ngram_range=(1, 2)):
        """
        Extrait les features TF-IDF du texte.
        
        Args:
            min_df (int): Fréquence minimale des termes
            max_df (float): Fréquence maximale des termes
            max_features (int): Nombre maximum de features
            ngram_range (tuple): Plage de n-grammes (min, max)
            
        Returns:
            scipy.sparse.csr_matrix: Matrice des features TF-IDF
        """
        print("Extraction des features TF-IDF...")
        
        # Création du vectoriseur
        self.vectorizer = TfidfVectorizer(
            preprocessor=self.preprocess_text,
            tokenizer=self.tokenize_and_stem,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            ngram_range=ngram_range,
            norm='l2',
            use_idf=True
        )
        
        # Extraction des features
        start_time = time.time()
        self.features = self.vectorizer.fit_transform(self.df[self.text_column])
        end_time = time.time()
        
        # Récupération des noms des features
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Extraction terminée en {end_time - start_time:.2f} secondes.")
        print(f"Nombre de features: {len(self.feature_names)}")
        print(f"Dimensions de la matrice: {self.features.shape}")
        
        return self.features
    
    def extract_bow_features(self, min_df=5, max_df=0.95, max_features=None, ngram_range=(1, 1)):
        """
        Extrait les features Bag-of-Words (Count) du texte.
        
        Args:
            min_df (int): Fréquence minimale des termes
            max_df (float): Fréquence maximale des termes
            max_features (int): Nombre maximum de features
            ngram_range (tuple): Plage de n-grammes (min, max)
            
        Returns:
            scipy.sparse.csr_matrix: Matrice des features BoW
        """
        print("Extraction des features Bag-of-Words...")
        
        # Création du vectoriseur
        self.vectorizer = CountVectorizer(
            preprocessor=self.preprocess_text,
            tokenizer=self.tokenize_and_stem,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            ngram_range=ngram_range
        )
        
        # Extraction des features
        start_time = time.time()
        self.features = self.vectorizer.fit_transform(self.df[self.text_column])
        end_time = time.time()
        
        # Récupération des noms des features
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Extraction terminée en {end_time - start_time:.2f} secondes.")
        print(f"Nombre de features: {len(self.feature_names)}")
        print(f"Dimensions de la matrice: {self.features.shape}")
        
        return self.features
    
    def visualize_feature_importance(self, top_n=20):
        """
        Visualise l'importance des features pour chaque classe.
        
        Args:
            top_n (int): Nombre de features importantes à afficher par classe
            
        Returns:
            plotly.graph_objects.Figure: Figure des features importantes
        """
        if self.features is None or self.feature_names is None:
            print("Les features n'ont pas été extraites. Appelez d'abord extract_tfidf_features() ou extract_bow_features().")
            return None
        
        # Récupération des étiquettes de classe
        y = self.df[self.class_column].values
        
        # Création d'une figure avec plusieurs sous-graphiques (un par classe)
        class_ids = sorted(self.df[self.class_column].unique())
        n_classes = len(class_ids)
        
        # Définir une palette de couleurs
        colors = px.colors.qualitative.Bold
        
        # Création de la figure
        fig = make_subplots(
            rows=int(np.ceil(n_classes/2)), 
            cols=2,
            subplot_titles=[f"Classe {c}" for c in class_ids]
        )
        
        # Pour chaque classe
        for i, class_id in enumerate(class_ids):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Filtre pour cette classe
            class_mask = (y == class_id)
            
            # Chi2 pour les features discriminantes
            chi2_selector = SelectKBest(chi2, k=top_n)
            chi2_selector.fit(self.features, class_mask)
            
            # Récupération des scores et indices
            scores = chi2_selector.scores_
            indices = np.argsort(scores)[::-1][:top_n]
            
            # Création des données pour le graphique
            feature_importance = []
            for j in indices:
                feature_importance.append({
                    'feature': self.feature_names[j],
                    'importance': scores[j]
                })
            
            # Conversion en DataFrame
            df_importance = pd.DataFrame(feature_importance)
            
            # Ajout du graphique
            fig.add_trace(
                go.Bar(
                    y=df_importance['feature'],
                    x=df_importance['importance'],
                    orientation='h',
                    name=f"Classe {class_id}",
                    marker_color=colors[i % len(colors)]
                ),
                row=row, col=col
            )
        
        # Mise à jour de la mise en page
        fig.update_layout(
            height=300 * int(np.ceil(n_classes/2)),
            width=1000,
            title_text="Features importantes par classe",
            showlegend=False
        )
        
        return fig

#===========================================================================================
# 3. Extraction de features structurelles
#===========================================================================================

class GraphFeatureExtractor:
    """Classe pour l'extraction de features structurelles à partir du graphe."""
    
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
            raise ValueError(f"Colonne ID '{id_column}' non trouvée dans le DataFrame.")
        
        # Attributs pour stocker les résultats
        self.graph = None
        self.features = None
        self.feature_names = None
    
    def build_graph(self, structural_columns, weights=None):
        """
        Construit un graphe à partir des colonnes structurelles.
        
        Args:
            structural_columns (dict): Dictionnaire des colonnes structurelles par type
            weights (dict): Poids à attribuer à chaque type de relation
            
        Returns:
            networkx.Graph: Graphe construit
        """
        print("Construction du graphe à partir des colonnes structurelles...")
        
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
                authors = getattr(row, authors_column) if hasattr(row, authors_column) else None
                
                if authors is None:
                    continue
                
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
                references = getattr(row, references_column) if hasattr(row, references_column) else None
                
                if references is None:
                    continue
                
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
                    if article1 in article_to_references and article2 in article_to_references:
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
                venue = getattr(row, venue_column) if hasattr(row, venue_column) else None
                
                if venue is not None and pd.notna(venue) and venue:
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
        
        print(f"Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
        
        # Stockage du graphe
        self.graph = G
        
        return G
    
    def extract_graph_features(self, feature_types=None):
        """
        Extrait les features structurelles à partir du graphe.
        
        Args:
            feature_types (list): Liste des types de features à extraire
            
        Returns:
            numpy.ndarray: Matrice des features structurelles
        """
        if self.graph is None:
            print("Le graphe n'a pas été construit. Appelez d'abord build_graph().")
            return None
        
        print("Extraction des features structurelles...")
        
        # Types de features par défaut
        if feature_types is None:
            feature_types = ['degree', 'centrality', 'clustering', 'pagerank']
        
        # Construction du mapping ID -> index
        id_to_index = {id_val: i for i, id_val in enumerate(self.df[self.id_column])}
        
        # Liste pour stocker les noms des features
        feature_names = []
        
        # Initialisation de la matrice de features (n_samples, 0)
        features = np.zeros((len(self.df), 0))
        
        # Extraction des features de degré
        if 'degree' in feature_types:
            print("Extraction des features de degré...")
            
            # Degré total
            degree = {node: val for node, val in self.graph.degree(weight='weight')}
            degree_features = np.zeros((len(self.df), 1))
            
            for node, val in degree.items():
                if node in id_to_index:
                    degree_features[id_to_index[node], 0] = val
            
            # Degré entrant (pour les graphes dirigés)
            if isinstance(self.graph, nx.DiGraph):
                in_degree = {node: val for node, val in self.graph.in_degree(weight='weight')}
                in_degree_features = np.zeros((len(self.df), 1))
                
                for node, val in in_degree.items():
                    if node in id_to_index:
                        in_degree_features[id_to_index[node], 0] = val
                
                # Degré sortant
                out_degree = {node: val for node, val in self.graph.out_degree(weight='weight')}
                out_degree_features = np.zeros((len(self.df), 1))
                
                for node, val in out_degree.items():
                    if node in id_to_index:
                        out_degree_features[id_to_index[node], 0] = val
                
                # Concaténation
                degree_features = np.hstack([degree_features, in_degree_features, out_degree_features])
                feature_names.extend(['degree', 'in_degree', 'out_degree'])
            else:
                feature_names.append('degree')
            
            # Ajout à la matrice de features
            features = np.hstack([features, degree_features])
        
        # Extraction des features de centralité
        if 'centrality' in feature_types:
            print("Extraction des features de centralité...")
            
            # Centralité de degré
            degree_centrality = nx.degree_centrality(self.graph)
            degree_centrality_features = np.zeros((len(self.df), 1))
            
            for node, val in degree_centrality.items():
                if node in id_to_index:
                    degree_centrality_features[id_to_index[node], 0] = val
            
            # Centralité d'intermédiarité (peut être coûteuse)
            betweenness_centrality = {}
            if self.graph.number_of_nodes() < 1000:
                betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight')
            else:
                # Approximation sur un échantillon
                betweenness_centrality = nx.betweenness_centrality(self.graph, k=100, weight='weight')
            
            betweenness_centrality_features = np.zeros((len(self.df), 1))
            
            for node, val in betweenness_centrality.items():
                if node in id_to_index:
                    betweenness_centrality_features[id_to_index[node], 0] = val
            
            # Centralité de proximité (peut être coûteuse)
            closeness_centrality = {}
            
            # Calcul uniquement sur la plus grande composante connexe
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            
            if subgraph.number_of_nodes() < 1000:
                closeness_centrality = nx.closeness_centrality(subgraph, distance='weight')
            
            closeness_centrality_features = np.zeros((len(self.df), 1))
            
            for node, val in closeness_centrality.items():
                if node in id_to_index:
                    closeness_centrality_features[id_to_index[node], 0] = val
            
            # Concaténation
            centrality_features = np.hstack([
                degree_centrality_features,
                betweenness_centrality_features,
                closeness_centrality_features
            ])
            
            feature_names.extend(['degree_centrality', 'betweenness_centrality', 'closeness_centrality'])
            
            # Ajout à la matrice de features
            features = np.hstack([features, centrality_features])
        
        # Extraction des features de clustering
        if 'clustering' in feature_types:
            print("Extraction des features de clustering...")
            
            # Coefficient de clustering
            clustering_coef = nx.clustering(self.graph, weight='weight')
            clustering_features = np.zeros((len(self.df), 1))
            
            for node, val in clustering_coef.items():
                if node in id_to_index:
                    clustering_features[id_to_index[node], 0] = val
            
            feature_names.append('clustering_coefficient')
            
            # Ajout à la matrice de features
            features = np.hstack([features, clustering_features])
        
        # Extraction des features de PageRank
        if 'pagerank' in feature_types:
            print("Extraction des features de PageRank...")
            
            # PageRank
            pagerank = nx.pagerank(self.graph, weight='weight')
            pagerank_features = np.zeros((len(self.df), 1))
            
            for node, val in pagerank.items():
                if node in id_to_index:
                    pagerank_features[id_to_index[node], 0] = val
            
            feature_names.append('pagerank')
            
            # Ajout à la matrice de features
            features = np.hstack([features, pagerank_features])
        
        # Stockage des résultats
        self.features = features
        self.feature_names = feature_names
        
        print(f"Extraction terminée. Dimensions de la matrice: {features.shape}")
        
        return features
    
    def visualize_graph_features(self):
        """
        Visualise la distribution des features structurelles.
        
        Returns:
            plotly.graph_objects.Figure: Figure de la distribution des features
        """
        if self.features is None or self.feature_names is None:
            print("Les features n'ont pas été extraites. Appelez d'abord extract_graph_features().")
            return None
        
        # Création de la figure
        fig = make_subplots(
            rows=int(np.ceil(len(self.feature_names)/2)), 
            cols=2,
            subplot_titles=self.feature_names
        )
        
        # Pour chaque feature
        for i, feature_name in enumerate(self.feature_names):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Extraction des valeurs
            values = self.features[:, i]
            
            # Histogramme
            fig.add_trace(
                go.Histogram(x=values, name=feature_name),
                row=row, col=col
            )
            
            # Mise à jour des axes
            fig.update_xaxes(title_text="Valeur", row=row, col=col)
            fig.update_yaxes(title_text="Fréquence", row=row, col=col)
        
        # Mise à jour de la mise en page
        fig.update_layout(
            height=300 * int(np.ceil(len(self.feature_names)/2)),
            width=900,
            title_text="Distribution des features structurelles",
            showlegend=False
        )
        
        return fig

#===========================================================================================
# 4. Combinaison des features et classification
#===========================================================================================

class DocumentClassifier:
    """Classe pour la classification des documents en combinant features textuelles et structurelles."""
    
    def __init__(self, df, text_features=None, graph_features=None, class_column='class'):
        """
        Initialise la classe avec le DataFrame d'articles et les features.
        
        Args:
            df (pandas.DataFrame): DataFrame contenant les articles
            text_features (scipy.sparse.csr_matrix): Matrice des features textuelles
            graph_features (numpy.ndarray): Matrice des features structurelles
            class_column (str): Nom de la colonne contenant les étiquettes de classe
        """
        self.df = df
        self.text_features = text_features
        self.graph_features = graph_features
        self.class_column = class_column
        
        # Vérification de la colonne de classe
        if class_column not in df.columns:
            raise ValueError(f"Colonne de classe '{class_column}' non trouvée dans le DataFrame.")
        
        # Récupération des étiquettes de classe
        self.y = df[class_column].values
        
        # Attributs pour stocker les résultats
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def combine_features(self, standardize_graph=True):
        """
        Combine les features textuelles et structurelles.
        
        Args:
            standardize_graph (bool): Standardiser les features structurelles
            
        Returns:
            scipy.sparse.csr_matrix: Matrice des features combinées
        """
        print("Combinaison des features textuelles et structurelles...")
        
        # Vérification des features
        if self.text_features is None and self.graph_features is None:
            raise ValueError("Aucune feature disponible. Veuillez fournir des features textuelles et/ou structurelles.")
        
        # Cas où seules les features textuelles sont disponibles
        if self.text_features is not None and self.graph_features is None:
            print("Utilisation uniquement des features textuelles.")
            self.X = self.text_features
            return self.X
        
        # Cas où seules les features structurelles sont disponibles
        if self.text_features is None and self.graph_features is not None:
            print("Utilisation uniquement des features structurelles.")
            
            # Standardisation si nécessaire
            if standardize_graph:
                scaler = StandardScaler()
                self.graph_features = scaler.fit_transform(self.graph_features)
            
            self.X = csr_matrix(self.graph_features)
            return self.X
        
        # Cas où les deux types de features sont disponibles
        if standardize_graph:
            scaler = StandardScaler()
            graph_features_scaled = scaler.fit_transform(self.graph_features)
        else:
            graph_features_scaled = self.graph_features
        
        # Conversion des features structurelles en matrice sparse
        graph_features_sparse = csr_matrix(graph_features_scaled)
        
        # Combinaison horizontale des matrices
        self.X = hstack([self.text_features, graph_features_sparse])
        
        print(f"Features combinées. Dimensions de la matrice: {self.X.shape}")
        
        return self.X
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test.
        
        Args:
            test_size (float): Proportion de l'ensemble de test
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X is None:
            raise ValueError("Les features n'ont pas été combinées. Appelez d'abord combine_features().")
        
        print(f"Division des données (test_size={test_size})...")
        
        # Division des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Ensemble d'entraînement: {self.X_train.shape}")
        print(f"Ensemble de test: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type='svm', **kwargs):
        """
        Entraîne un modèle de classification.
        
        Args:
            model_type (str): Type de modèle ('svm', 'rf', 'lr', 'nb', 'knn', 'gb', 'mlp')
            **kwargs: Paramètres spécifiques au modèle
            
        Returns:
            object: Modèle entraîné
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Les données n'ont pas été divisées. Appelez d'abord split_data().")
        
        print(f"Entraînement d'un modèle {model_type}...")
        
        # Création du modèle
        if model_type == 'svm':
            model = SVC(probability=True, **kwargs)
        elif model_type == 'linear_svm':
            model = LinearSVC(**kwargs)
        elif model_type == 'rf':
            model = RandomForestClassifier(**kwargs)
        elif model_type == 'lr':
            model = LogisticRegression(**kwargs)
        elif model_type == 'nb':
            model = MultinomialNB(**kwargs)
        elif model_type == 'knn':
            model = KNeighborsClassifier(**kwargs)
        elif model_type == 'gb':
            model = GradientBoostingClassifier(**kwargs)
        elif model_type == 'mlp':
            model = MLPClassifier(**kwargs)
        else:
            raise ValueError(f"Type de modèle '{model_type}' non reconnu.")
        
        # Entraînement du modèle
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")
        
        # Stockage du modèle
        self.models[model_type] = model
        
        return model
    
    def evaluate_model(self, model_type=None, model=None):
        """
        Évalue un modèle sur l'ensemble de test.
        
        Args:
            model_type (str): Type de modèle à évaluer
            model (object): Modèle à évaluer (si model_type n'est pas spécifié)
            
        Returns:
            dict: Résultats de l'évaluation
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Les données n'ont pas été divisées. Appelez d'abord split_data().")
        
        if model is None:
            if model_type is None or model_type not in self.models:
                raise ValueError(f"Type de modèle '{model_type}' non disponible. Entraînez d'abord le modèle avec train_model().")
            
            model = self.models[model_type]
        else:
            model_type = model.__class__.__name__
        
        print(f"Évaluation du modèle {model_type}...")
        
        # Prédictions sur l'ensemble de test
        y_pred = model.predict(self.X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        print(f"Exactitude (accuracy): {accuracy:.4f}")
        print("\nRapport de classification:")
        print(classification_report(self.y_test, y_pred))
        
        # Stockage des résultats
        results = {
            'model_type': model_type,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred
        }
        
        self.results[model_type] = results
        
        return results
    
    def train_and_evaluate_multiple_models(self, models_config=None):
        """
        Entraîne et évalue plusieurs modèles.
        
        Args:
            models_config (list): Liste de configurations de modèles
            
        Returns:
            dict: Résultats de l'évaluation pour chaque modèle
        """
        if models_config is None:
            # Configuration par défaut
            models_config = [
                {'type': 'svm', 'params': {'C': 1.0, 'kernel': 'linear', 'random_state': 42}},
                {'type': 'rf', 'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}},
                {'type': 'lr', 'params': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}},
                {'type': 'nb', 'params': {'alpha': 0.1}},
                {'type': 'knn', 'params': {'n_neighbors': 5, 'weights': 'distance'}}
            ]
        
        # Pour chaque configuration
        for config in models_config:
            model_type = config['type']
            params = config['params']
            
            # Entraînement du modèle
            model = self.train_model(model_type=model_type, **params)
            
            # Évaluation du modèle
            self.evaluate_model(model=model)
        
        # Affichage comparatif
        print("\nComparaison des modèles:")
        for model_type, results in self.results.items():
            print(f"{model_type}: Accuracy = {results['accuracy']:.4f}")
        
        return self.results
    
    def optimize_hyperparameters(self, model_type='svm', param_grid=None, cv=5, scoring='accuracy'):
        """
        Optimise les hyperparamètres d'un modèle.
        
        Args:
            model_type (str): Type de modèle à optimiser
            param_grid (dict): Grille de paramètres à tester
            cv (int): Nombre de plis pour la validation croisée
            scoring (str): Métrique pour l'évaluation
            
        Returns:
            dict: Meilleurs paramètres et résultats
        """
        if self.X is None or self.y is None:
            raise ValueError("Les features n'ont pas été combinées. Appelez d'abord combine_features().")
        
        print(f"Optimisation des hyperparamètres pour le modèle {model_type}...")
        
        # Grilles de paramètres par défaut
        if param_grid is None:
            if model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1]
                }
            elif model_type == 'linear_svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'dual': [True, False],
                    'max_iter': [1000, 2000]
                }
            elif model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == 'lr':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga'],
                    'penalty': ['l1', 'l2']
                }
            elif model_type == 'nb':
                param_grid = {
                    'alpha': [0.01, 0.1, 0.5, 1.0]
                }
            elif model_type == 'knn':
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1 = manhattan, 2 = euclidean
                }
            elif model_type == 'gb':
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            elif model_type == 'mlp':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            else:
                raise ValueError(f"Type de modèle '{model_type}' non reconnu.")
        
        # Création du modèle de base
        if model_type == 'svm':
            model = SVC(probability=True)
        elif model_type == 'linear_svm':
            model = LinearSVC()
        elif model_type == 'rf':
            model = RandomForestClassifier()
        elif model_type == 'lr':
            model = LogisticRegression()
        elif model_type == 'nb':
            model = MultinomialNB()
        elif model_type == 'knn':
            model = KNeighborsClassifier()
        elif model_type == 'gb':
            model = GradientBoostingClassifier()
        elif model_type == 'mlp':
            model = MLPClassifier()
        else:
            raise ValueError(f"Type de modèle '{model_type}' non reconnu.")
        
        # Création de la recherche sur grille
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Entraînement du modèle
        start_time = time.time()
        grid_search.fit(self.X, self.y)
        end_time = time.time()
        
        print(f"Optimisation terminée en {end_time - start_time:.2f} secondes.")
        print(f"Meilleurs paramètres: {grid_search.best_params_}")
        print(f"Meilleur score: {grid_search.best_score_:.4f}")
        
        # Entraînement du modèle avec les meilleurs paramètres
        self.train_model(model_type=model_type, **grid_search.best_params_)
        
        # Stockage des résultats
        results = {
            'model_type': model_type,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return results
    
    def visualize_confusion_matrix(self, model_type=None, normalize=True):
        """
        Visualise la matrice de confusion.
        
        Args:
            model_type (str): Type de modèle à visualiser
            normalize (bool): Normaliser la matrice de confusion
            
        Returns:
            plotly.graph_objects.Figure: Figure de la matrice de confusion
        """
        if model_type is None:
            if not self.results:
                raise ValueError("Aucun modèle évalué. Appelez d'abord evaluate_model().")
            
            # Utiliser le premier modèle disponible
            model_type = list(self.results.keys())[0]
        
        if model_type not in self.results:
            raise ValueError(f"Type de modèle '{model_type}' non évalué. Appelez d'abord evaluate_model().")
        
        # Récupération de la matrice de confusion
        conf_matrix = self.results[model_type]['confusion_matrix']
        
        # Normalisation si nécessaire
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            conf_matrix = np.round(conf_matrix, 2)
        
        # Classes uniques
        class_ids = sorted(np.unique(self.y))
        
        # Mapping des classes
        class_mapping = {
            1: "Artificial Intelligence",
            2: "Data Science",
            3: "Interface",
            4: "Computer Vision",
            5: "Network",
            6: "Theoretical CS",
            7: "Specific Applications",
            8: "Other"
        }
        
        class_names = [class_mapping.get(c, f"Class {c}") for c in class_ids]
        
        # Création de la figure
        fig = px.imshow(
            conf_matrix,
            x=class_names,
            y=class_names,
            color_continuous_scale='viridis',
            labels=dict(x="Prédiction", y="Vraie classe", color="Proportion" if normalize else "Nombre"),
            title=f"Matrice de confusion - {model_type}"
        )
        
        # Ajout des annotations
        annotations = []
        for i, row in enumerate(conf_matrix):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(value),
                        showarrow=False,
                        font=dict(color='white' if value > 0.5 else 'black')
                    )
                )
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def visualize_feature_importance(self, model_type='rf', top_n=20):
        """
        Visualise l'importance des features pour un modèle.
        
        Args:
            model_type (str): Type de modèle ('rf', 'gb' ou 'linear_svm')
            top_n (int): Nombre de features importantes à afficher
            
        Returns:
            plotly.graph_objects.Figure: Figure de l'importance des features
        """
        if model_type not in self.models:
            raise ValueError(f"Type de modèle '{model_type}' non disponible. Entraînez d'abord le modèle avec train_model().")
        
        model = self.models[model_type]
        
        # Vérification du type de modèle
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            print(f"Le modèle {model_type} ne fournit pas d'importance des features.")
            return None
        
        # Extraction de l'importance des features
        if hasattr(model, 'feature_importances_'):
            # Pour RandomForest, GradientBoosting, etc.
            feature_importance = model.feature_importances_
        else:
            # Pour les modèles linéaires (LinearSVC, LogisticRegression, etc.)
            feature_importance = np.abs(model.coef_).mean(axis=0)
        
        # Récupération des noms des features
        feature_names = []
        
        if hasattr(self, 'text_features') and self.text_features is not None:
            # Features textuelles
            if hasattr(self, 'text_feature_extractor') and hasattr(self.text_feature_extractor, 'feature_names'):
                feature_names.extend(self.text_feature_extractor.feature_names)
            else:
                feature_names.extend([f'text_{i}' for i in range(self.text_features.shape[1])])
        
        if hasattr(self, 'graph_features') and self.graph_features is not None:
            # Features structurelles
            if hasattr(self, 'graph_feature_extractor') and hasattr(self.graph_feature_extractor, 'feature_names'):
                feature_names.extend(self.graph_feature_extractor.feature_names)
            else:
                feature_names.extend([f'graph_{i}' for i in range(self.graph_features.shape[1])])
        
        # S'assurer que le nombre de noms correspond au nombre de features
        if len(feature_names) != len(feature_importance):
            print(f"Attention: Le nombre de noms de features ({len(feature_names)}) ne correspond pas au nombre de features ({len(feature_importance)}).")
            feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
        
        # Création d'un DataFrame pour la visualisation
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        
        # Tri par importance décroissante
        df_importance = df_importance.sort_values('importance', ascending=False).head(top_n)
        
        # Création de la figure
        fig = px.bar(
            df_importance,
            x='importance',
            y='feature',
            orientation='h',
            labels={'importance': 'Importance', 'feature': 'Feature'},
            title=f"Importance des features - {model_type}"
        )
        
        fig.update_layout(height=600, width=800)
        
        return fig

# Création de la colonne 'class' à partir de la colonne 'venue'
def assign_class_from_venue(venue_name):
    """Assigne une classe (1-8) en fonction du nom de la conférence/journal"""
    venue_lower = str(venue_name).lower() if pd.notna(venue_name) else ""
    
    # Classification selon les critères donnés dans l'énoncé
    if any(term in venue_lower for term in ['machine learning', 'artificial intelligence', 'neural', 
                                           'autonomous', 'agent', 'nlp', 'natural language']):
        return 1  # Artificial Intelligence
    elif any(term in venue_lower for term in ['data', 'information system', 'database', 'mining', 
                                             'cleaning', 'business intelligence']):
        return 2  # Data Science
    elif any(term in venue_lower for term in ['visualization', 'interface', 'interaction', 'hci']):
        return 3  # Interface
    elif any(term in venue_lower for term in ['vision', 'image', '2d', '3d', 'virtual reality']):
        return 4  # Computer Vision
    elif any(term in venue_lower for term in ['network', 'system', 'security', 'mobile', 'iot', 'web']):
        return 5  # Network
    elif any(term in venue_lower for term in ['theory', 'theorem', 'proof', 'bound', 
                                             'calculability', 'compilation', 'game theory']):
        return 6  # Theoretical CS
    elif any(term in venue_lower for term in ['humanities', 'biology', 'medicine', 'chemistry',
                                             'physics', 'social']):
        return 7  # Specific Applications
    else:
        return 8  # Other
