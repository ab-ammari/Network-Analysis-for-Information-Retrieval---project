# Author : Ammari Abdelhafid
# Exercice 1 : Acquisition des données
# M2 MIASHS : projet Network Analysis for Information Retrieval

# Importation des bibliothèques nécessaires
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import re
from datetime import datetime

# Configuration pour afficher plus de colonnes
pd.set_option('display.max_columns', None)


# 1. Chargement des données
def load_citation_network_data_random(file_path, percentage=100, random_seed=42):
    """
    Charge un pourcentage aléatoire des données du Citation Network Dataset.
    
    Args:
        file_path (str): Chemin vers le fichier JSON
        percentage (float): Pourcentage des données à charger (entre 0 et 100)
        random_seed (int): Graine pour la reproductibilité
        
    Returns:
        list: Liste des articles sous forme de dictionnaires
    """
    articles = []
    
    if percentage <= 0 or percentage > 100:
        print(f"Le pourcentage doit être entre 0 et 100. Utilisation de 100% par défaut.")
        percentage = 100
    
    try:
        # D'abord, compter le nombre total de lignes dans le fichier
        total_lines = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        
        # Calculer combien de lignes charger
        lines_to_load = int(total_lines * percentage / 100)
        
        print(f"Fichier contient {total_lines} articles. Chargement de {percentage}% ({lines_to_load} articles)...")
        
        # Sélectionner les indices de lignes à charger aléatoirement
        np.random.seed(random_seed)
        selected_indices = set(np.random.choice(total_lines, lines_to_load, replace=False))
        
        # Charger les lignes sélectionnées
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f), desc=f"Chargement de {percentage}% des articles", total=total_lines):
                if i in selected_indices:
                    try:
                        article = json.loads(line.strip())
                        articles.append(article)
                    except json.JSONDecodeError:
                        print(f"Erreur de décodage JSON pour la ligne: {line[:50]}...")
                        continue
        
        print(f"Nombre total d'articles chargés: {len(articles)} ({len(articles)/total_lines*100:.1f}% du fichier)")
        return articles
    
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'a pas été trouvé.")
        return []
    
# 2. Exploration des données
def explore_dataset(articles):
    """
    Explore la structure du jeu de données et affiche des informations générales.
    
    Args:
        articles (list): Liste des articles
    """
    if len(articles) == 0:
        print("Aucun article à explorer.")
        return
    
    # Afficher la structure d'un article
    print("\n2.1. Structure d'un article d'exemple:")
    sample_article = articles[0]
    for key, value in sample_article.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"{key}: {value[:3]} ... (total: {len(value)})")
        else:
            print(f"{key}: {value}")
    
    # Vérifier les clés disponibles
    print("\n2.2. Clés disponibles dans le jeu de données:")
    all_keys = set()
    for article in articles[:1000]:  # Analyse des 1000 premiers pour performance
        all_keys.update(article.keys())
    print(all_keys)
    
    # Calculer la disponibilité des champs
    print("\n2.3. Disponibilité des champs:")
    field_availability = {}
    for key in all_keys:
        count = sum(1 for article in articles if key in article)
        field_availability[key] = (count, count/len(articles)*100)
    
    for key, (count, percentage) in field_availability.items():
        print(f"{key}: {count} articles ({percentage:.2f}%)")

# 3. Conversion en DataFrame
def convert_to_dataframe(articles):
    """
    Convertit la liste d'articles en DataFrame pandas pour faciliter l'analyse.
    
    Args:
        articles (list): Liste des articles
        
    Returns:
        pandas.DataFrame: DataFrame contenant les articles
    """
    # Sélection des champs pertinents
    processed_articles = []
    
    for article in tqdm(articles, desc="Conversion en DataFrame"):
        processed_article = {
            'id': article.get('id', None),
            'title': article.get('title', ''),
            'abstract': article.get('abstract', ''),
            'year': article.get('year', None),
            'venue': article.get('venue', ''),  # Conférence ou journal
            'n_citation': article.get('n_citation', 0),  # Nombre de citations
            'author_count': len(article.get('authors', [])),
            'references_count': len(article.get('references', [])),
        }
        
        # Traitement des auteurs (déjà sous forme de chaînes de caractères dans ce jeu de données)
        authors = article.get('authors', [])
        # Les auteurs sont déjà des chaînes de caractères, pas des dictionnaires
        processed_article['authors'] = authors
        processed_article['author_names'] = ', '.join(authors) if authors else ''
        
        # Stockage des références
        processed_article['references'] = article.get('references', [])
        
        processed_articles.append(processed_article)
    
    df = pd.DataFrame(processed_articles)
    return df

# 4. Analyse statistique de base
def basic_statistics(df):
    """
    Calcule et affiche des statistiques de base sur le jeu de données.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les articles
    """
    print("\n4.1. Informations générales sur le DataFrame:")
    print(df.info())
    
    print("\n4.2. Statistiques descriptives:")
    print(df.describe())
    
    # Distribution par année (devrait être principalement 2015)
    print("\n4.3. Distribution par année:")
    year_counts = df['year'].value_counts().sort_index()
    print(year_counts)
    
    # Top venues (conférences/journaux)
    print("\n4.4. Top 10 des conférences/journaux:")
    venue_counts = df['venue'].value_counts().head(10)
    print(venue_counts)
    
    # Distribution du nombre d'auteurs par article
    print("\n4.5. Statistiques sur le nombre d'auteurs par article:")
    print(df['author_count'].describe())
    
    # Distribution du nombre de références par article
    print("\n4.6. Statistiques sur le nombre de références par article:")
    print(df['references_count'].describe())
    
    # Distribution du nombre de citations reçues
    print("\n4.7. Statistiques sur le nombre de citations reçues:")
    print(df['n_citation'].describe())

# 5. Visualisations de base
def basic_visualizations(df):
    """
    Crée et affiche des visualisations de base pour explorer le jeu de données.
    Adaptée pour l'affichage direct dans Jupyter Notebook.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les articles
    """
    # Configuration pour un meilleur affichage dans Jupyter
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Figure 1: Statistiques générales
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 5.1. Distribution du nombre d'auteurs par article
    sns.histplot(df['author_count'].clip(0, 10), kde=True, bins=10, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution du nombre d\'auteurs par article', fontsize=12)
    axes[0, 0].set_xlabel('Nombre d\'auteurs')
    axes[0, 0].set_ylabel('Nombre d\'articles')
    
    # 5.2. Distribution du nombre de références par article
    sns.histplot(df['references_count'].clip(0, 50), kde=True, bins=20, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution du nombre de références par article', fontsize=12)
    axes[0, 1].set_xlabel('Nombre de références')
    axes[0, 1].set_ylabel('Nombre d\'articles')
    
    # 5.3. Distribution du nombre de citations
    sns.histplot(df['n_citation'].clip(0, 50), kde=True, bins=20, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution du nombre de citations reçues', fontsize=12)
    axes[1, 0].set_xlabel('Nombre de citations')
    axes[1, 0].set_ylabel('Nombre d\'articles')
    
    # 5.4. Top 10 des conférences/journaux
    top_venues = df['venue'].value_counts().head(10)
    sns.barplot(x=top_venues.values, y=top_venues.index, ax=axes[1, 1])
    axes[1, 1].set_title('Top 10 des conférences/journaux', fontsize=12)
    axes[1, 1].set_xlabel('Nombre d\'articles')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Auteurs les plus prolifiques
    plt.figure(figsize=(12, 8))
    all_authors = [author for sublist in df['authors'] for author in sublist]
    top_authors = Counter(all_authors).most_common(20)
    author_df = pd.DataFrame(top_authors, columns=['Auteur', 'Nombre d\'articles'])
    sns.barplot(x='Nombre d\'articles', y='Auteur', data=author_df)
    plt.title('Top 20 des auteurs les plus prolifiques', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("Visualisations affichées directement dans le notebook")

# 6. Enregistrement des données traitées
def save_processed_data(df, output_file):
    """
    Enregistre les données traitées pour une utilisation ultérieure.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les articles traités
        output_file (str): Chemin du fichier de sortie
    """
    print(f"\nEnregistrement des données traitées dans {output_file}")
    df.to_pickle(output_file)
    print(f"Données enregistrées avec succès. Shape: {df.shape}")
