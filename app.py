import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import numpy as np
from datetime import datetime, timedelta
import json

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Agence de Livraison - Dashboard Neo4j",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'esthétique
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: white;
        font-size: 1.3rem;
        opacity: 0.9;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric > div > div:first-child {
        font-size: 0.9rem;
        color: #555;
    }
    .stMetric > div > div:nth-child(2) {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
    }
    .stMetric > div > div:nth-child(3) {
        font-size: 0.8rem;
        color: #777;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.1rem;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 1rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #5a6ad0;
    }
</style>
""", unsafe_allow_html=True)

# Classe pour la connexion Neo4j
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

# Configuration de la connexion Neo4j (mise en cache pour éviter les reconnexions)
@st.cache_resource
def init_neo4j_connection():
    # Ces informations devraient idéalement être dans des variables d'environnement
    # ou un fichier de configuration sécurisé.
    URI = st.secrets["neo4j"]["uri"] if "neo4j" in st.secrets else "bolt://localhost:7687"
    USER = st.secrets["neo4j"]["user"] if "neo4j" in st.secrets else "neo4j"
    PASSWORD = st.secrets["neo4j"]["password"] if "neo4j" in st.secrets else "password"
    
    try:
        connection = Neo4jConnection(URI, USER, PASSWORD)
        # Test de la connexion
        connection.execute_query("RETURN 1")
        return connection
    except Exception as e:
        st.error(f"❌ Erreur de connexion à Neo4j: {e}. Veuillez vérifier l'URI, l'utilisateur et le mot de passe.")
        st.stop() # Arrête l'exécution de l'application si la connexion échoue
        return None

# Header principal de l'application
st.markdown("""
<div class="main-header">
    <h1>🚚 Agence de Livraison - Dashboard Neo4j</h1>
    <p>Système de gestion et d'optimisation des flux logistiques</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choisissez une section",
    [
        "🏠 Tableau de Bord",
        "📍 Analyse des Zones",
        "🛣️ Optimisation Trajets",
        "👥 Gestion Livreurs",
        "📦 Analyse Produits",
        "👤 Gestion Clients",
        "📊 Reporting Exécutif",
        "🔧 Administration"
    ]
)

# Initialisation de la connexion Neo4j
conn = init_neo4j_connection()

# Fonction utilitaire pour exécuter les requêtes en toute sécurité
def execute_safe_query(query, title="Requête"):
    if conn is None:
        st.error("La connexion à Neo4j n'est pas établie.")
        return []
    try:
        result = conn.execute_query(query)
        return result
    except Exception as e:
        st.error(f"⚠️ Erreur lors de l'exécution de la requête '{title}': {e}")
        return []

# ==============================================================================
# SECTION : TABLEAU DE BORD
# ==============================================================================
if page == "🏠 Tableau de Bord":
    st.header("📊 Vue d'ensemble de l'agence")
    
    st.markdown("---")

    # KPIs principaux
    st.subheader("📈 Indicateurs Clés de Performance")
    kpi_query = """
    MATCH (cmd:Commande)
    WITH COUNT(cmd) as total_commandes, SUM(cmd.prix_total) as ca_total, 
         AVG(cmd.prix_total) as panier_moyen, SUM(cmd.poids_total) as poids_total
    MATCH (l:Livreur)
    WITH total_commandes, ca_total, panier_moyen, poids_total, COUNT(l) as nb_livreurs
    MATCH (c:Client)
    WITH total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, COUNT(c) as nb_clients
    MATCH (z:Zone)
    WITH total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, nb_clients, COUNT(z) as nb_zones
    MATCH (e:Entrepôt)
    WITH total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, nb_clients, nb_zones, COUNT(e) as nb_entrepots
    RETURN total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, nb_clients, nb_zones, nb_entrepots
    """
    
    kpi_data = execute_safe_query(kpi_query, "KPIs")
    
    if kpi_data:
        kpi = kpi_data[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Commandes", kpi.get('total_commandes', 0))
            st.metric("Nombre Clients", kpi.get('nb_clients', 0))
        
        with col2:
            st.metric("CA Total", f"{kpi.get('ca_total', 0):,.2f} DH")
            st.metric("Nombre Livreurs", kpi.get('nb_livreurs', 0))
        
        with col3:
            st.metric("Panier Moyen", f"{kpi.get('panier_moyen', 0):,.2f} DH")
            st.metric("Nombre Zones", kpi.get('nb_zones', 0))
        
        with col4:
            st.metric("Poids Total", f"{kpi.get('poids_total', 0):,.2f} kg")
            st.metric("Nombre Entrepôts", kpi.get('nb_entrepots', 0))
    else:
        st.info("Aucune donnée KPI disponible. La base de données est peut-être vide ou la requête a échoué.")

    st.markdown("---")

    # Graphiques de synthèse
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Évolution des Commandes")
        evolution_query = """
        MATCH (cmd:Commande)
        RETURN cmd.date_commande as date, COUNT(cmd) as nb_commandes, 
             SUM(cmd.prix_total) as ca_jour
        ORDER BY date
        """
        
        evolution_data = execute_safe_query(evolution_query, "Évolution des Commandes")
        if evolution_data:
            df_evolution = pd.DataFrame(evolution_data)
            df_evolution['date'] = pd.to_datetime(df_evolution['date']) # Convertir en datetime
            fig_evolution = px.line(df_evolution, x='date', y='nb_commandes', 
                                  title="Nombre de commandes par jour",
                                  labels={'date': 'Date', 'nb_commandes': 'Nombre de Commandes'})
            st.plotly_chart(fig_evolution, use_container_width=True)
        else:
            st.info("Aucune donnée d'évolution des commandes disponible.")
    
    with col2:
        st.subheader("🏆 Top Zones par CA")
        top_zones_query = """
        MATCH (c:Client)-[:LOCATED_IN]->(z:Zone), (c)-[:ORDERED]->(cmd:Commande)
        RETURN z.nom as zone, SUM(cmd.prix_total) as ca_total, COUNT(cmd) as nb_commandes
        ORDER BY ca_total DESC
        LIMIT 5
        """
        
        zones_data = execute_safe_query(top_zones_query, "Top Zones par CA")
        if zones_data:
            df_zones = pd.DataFrame(zones_data)
            fig_zones = px.bar(df_zones, x='zone', y='ca_total', 
                             title="Top 5 Zones par Chiffre d'Affaires",
                             labels={'zone': 'Zone', 'ca_total': 'Chiffre d\'Affaires (DH)'})
            st.plotly_chart(fig_zones, use_container_width=True)
        else:
            st.info("Aucune donnée de top zones disponible.")
    
    st.markdown("---")

    # Statut des commandes
    st.subheader("📋 Statut des Commandes")
    statut_query = """
    MATCH (cmd:Commande)
    RETURN cmd.statut as statut, COUNT(cmd) as nb_commandes, 
         SUM(cmd.prix_total) as ca_statut
    ORDER BY nb_commandes DESC
    """
    
    statut_data = execute_safe_query(statut_query, "Statut des Commandes")
    if statut_data:
        df_statut = pd.DataFrame(statut_data)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(df_statut, values='nb_commandes', names='statut',
                           title="Répartition des commandes par statut")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(df_statut, x='statut', y='ca_statut',
                           title="Chiffre d'affaires par statut",
                           labels={'statut': 'Statut', 'ca_statut': 'Chiffre d\'Affaires (DH)'})
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Aucune donnée de statut de commande disponible.")

# ==============================================================================
# SECTION : ANALYSE DES ZONES
# ==============================================================================
elif page == "📍 Analyse des Zones":
    st.header("🗺️ Analyse des Zones de Livraison")
    
    st.markdown("---")

    # Zones à forte densité
    st.subheader("🎯 Zones à Forte Densité de Commandes")
    
    zones_query = """
    MATCH (c:Client)-[:LOCATED_IN]->(z:Zone)<-[:ASSIGNED_TO]-(l:Livreur)-[:DELIVERS]->(cmd:Commande)
    RETURN z.nom as Zone, z.densite_population as Densité, COUNT(cmd) as Commandes,
           COUNT(DISTINCT c) as Clients, COUNT(DISTINCT l) as Livreurs, SUM(cmd.prix_total) as CA_Zone,
           CASE WHEN COUNT(DISTINCT l) > 0 THEN ROUND(COUNT(cmd) * 1.0 / COUNT(DISTINCT l), 2) ELSE 0 END as Charge_par_Livreur
    ORDER BY Commandes DESC
    """
    
    zones_data = execute_safe_query(zones_query, "Zones à Forte Densité")
    if zones_data:
        df_zones = pd.DataFrame(zones_data)
        
        st.dataframe(df_zones, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(df_zones, x='Commandes', y='CA_Zone', 
                                   size='Clients', color='Densité',
                                   hover_data=['Zone', 'Livreurs'],
                                   title="Relation Commandes vs CA par Zone",
                                   labels={'Commandes': 'Nombre de Commandes', 'CA_Zone': 'CA de la Zone (DH)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(df_zones, x='Zone', y='Charge_par_Livreur',
                           color='Charge_par_Livreur',
                           title="Charge de travail par livreur par zone",
                           labels={'Zone': 'Zone', 'Charge_par_Livreur': 'Commandes par Livreur'})
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Aucune donnée de zone disponible.")
    
    st.markdown("---")

    # Analyse des goulets d'étranglement
    st.subheader("⚠️ Détection des Goulets d'Étranglement")
    
    goulets_query = """
    MATCH (z:Zone)<-[:ASSIGNED_TO]-(l:Livreur)-[:DELIVERS]->(cmd:Commande)
    WITH z, COUNT(DISTINCT l) as nb_livreurs, COUNT(cmd) as nb_commandes,
         CASE WHEN COUNT(DISTINCT l) > 0 THEN ROUND(COUNT(cmd) * 1.0 / COUNT(DISTINCT l), 2) ELSE 0 END as charge_par_livreur
    WHERE nb_livreurs > 0
    RETURN z.nom as Zone, z.densite_population as Densité,
           nb_livreurs as Nb_Livreurs, nb_commandes as Nb_Commandes,
           charge_par_livreur as Commandes_par_Livreur,
           CASE 
             WHEN charge_par_livreur > 20 THEN 'Surchargée'
             WHEN charge_par_livreur > 10 THEN 'Normale'
             ELSE 'Sous-utilisée'
           END as État_Charge
    ORDER BY charge_par_livreur DESC
    """
    
    goulets_data = execute_safe_query(goulets_query, "Goulets d'Étranglement")
    if goulets_data:
        df_goulets = pd.DataFrame(goulets_data)
        
        def color_status(val):
            if val == 'Surchargée':
                return 'background-color: #ffebee; color: #c62828' # Rouge clair
            elif val == 'Normale':
                return 'background-color: #e8f5e8; color: #2e7d32' # Vert clair
            else:
                return 'background-color: #e3f2fd; color: #1565c0' # Bleu clair
        
        styled_df = df_goulets.style.applymap(color_status, subset=['État_Charge'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Aucune donnée de goulets d'étranglement disponible.")

# ==============================================================================
# SECTION : OPTIMISATION TRAJETS
# ==============================================================================
elif page == "🛣️ Optimisation Trajets":
    st.header("🚀 Optimisation des Itinéraires")
    
    st.markdown("---")

    # Analyse des trajets coûteux
    st.subheader("💰 Analyse des Coûts de Transport")
    
    trajets_query = """
    MATCH (t:Trajet)
    OPTIONAL MATCH (t)-[:PASSED_BY]->(z:Zone)
    WITH t, z, 
         CASE WHEN t.distance > 0 THEN ROUND(t.cout / t.distance, 2) ELSE 0 END as cout_par_km,
         CASE WHEN t.duree > 0 THEN ROUND(t.distance / t.duree * 60, 2) ELSE 0 END as vitesse_kmh
    RETURN t.id as Trajet, t.origine as Origine, COALESCE(z.nom, 'N/A') as Destination, 
           t.distance as Distance_km, t.duree as Durée_min, t.cout as Coût_DH,
           cout_par_km as Coût_par_km, vitesse_kmh as Vitesse_kmh,
           CASE 
             WHEN vitesse_kmh < 20 AND vitesse_kmh > 0 THEN 'Lent'
             WHEN vitesse_kmh >= 20 AND vitesse_kmh < 40 THEN 'Moyen'
             WHEN vitesse_kmh >= 40 THEN 'Rapide'
             ELSE 'N/A'
           END as Performance
    ORDER BY cout_par_km DESC
    """
    
    trajets_data = execute_safe_query(trajets_query, "Analyse des Trajets")
    if trajets_data:
        df_trajets = pd.DataFrame(trajets_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Distance Totale", f"{df_trajets['Distance_km'].sum():.1f} km")
        with col2:
            st.metric("Coût Total", f"{df_trajets['Coût_DH'].sum():.2f} DH")
        with col3:
            st.metric("Vitesse Moyenne", f"{df_trajets['Vitesse_kmh'].mean():.1f} km/h")
        with col4:
            st.metric("Coût Moyen/km", f"{df_trajets['Coût_par_km'].mean():.2f} DH")
        
        st.dataframe(df_trajets, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(df_trajets, x='Distance_km', y='Coût_DH',
                                   size='Vitesse_kmh', color='Performance',
                                   hover_data=['Trajet', 'Origine', 'Destination'],
                                   title="Relation Distance vs Coût",
                                   labels={'Distance_km': 'Distance (km)', 'Coût_DH': 'Coût (DH)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_box = px.box(df_trajets, x='Performance', y='Coût_par_km',
                           title="Distribution des coûts par performance",
                           labels={'Performance': 'Performance', 'Coût_par_km': 'Coût par km (DH)'})
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Aucune donnée de trajet disponible.")
    
    st.markdown("---")

    # Optimisation des itinéraires
    st.subheader("🎯 Suggestions d'Optimisation")
    
    optim_query = """
    MATCH (e:Entrepôt), (t:Trajet {origine: e.id})-[:PASSED_BY]->(z:Zone)
    WITH e, z, MIN(t.cout) as cout_min, MIN(t.duree) as duree_min, MIN(t.distance) as distance_min
    RETURN e.nom as Entrepôt, z.nom as Zone_Destination,
           distance_min as Distance_Optimale_km, duree_min as Durée_Optimale_min,
           cout_min as Coût_Optimal_DH
    ORDER BY e.nom, cout_min
    """
    
    optim_data = execute_safe_query(optim_query, "Optimisation des Itinéraires")
    if optim_data:
        df_optim = pd.DataFrame(optim_data)
        st.dataframe(df_optim, use_container_width=True)
        
        fig_heatmap = px.density_heatmap(df_optim, x='Entrepôt', y='Zone_Destination', 
                                       z='Coût_Optimal_DH', title="Matrice des coûts optimaux",
                                       labels={'Entrepôt': 'Entrepôt', 'Zone_Destination': 'Zone de Destination', 'Coût_Optimal_DH': 'Coût Optimal (DH)'})
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Aucune suggestion d'optimisation disponible.")

# ==============================================================================
# SECTION : GESTION LIVREURS
# ==============================================================================
elif page == "👥 Gestion Livreurs":
    st.header("👨‍💼 Gestion des Livreurs")
    
    st.markdown("---")

    # Performance des livreurs
    st.subheader("📊 Performance des Livreurs")
    
    livreurs_query = """
    MATCH (l:Livreur)-[:ASSIGNED_TO]->(z:Zone), (l)-[:DELIVERS]->(cmd:Commande)
    WITH l, z, COUNT(cmd) as nb_livraisons, SUM(cmd.prix_total) as ca_genere,
         CASE WHEN COUNT(cmd) > 0 THEN ROUND(SUM(cmd.prix_total) / COUNT(cmd), 2) ELSE 0 END as ca_moyen
    RETURN l.nom as Livreur, l.vehicule as Véhicule, l.experience as Expérience,
           z.nom as Zone, nb_livraisons as Livraisons, ca_genere as CA_Généré,
           ca_moyen as CA_Moyen_Livraison
    ORDER BY nb_livraisons DESC
    """
    
    livreurs_data = execute_safe_query(livreurs_query, "Performance des Livreurs")
    if livreurs_data:
        df_livreurs = pd.DataFrame(livreurs_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Livraisons", df_livreurs['Livraisons'].sum())
        with col2:
            st.metric("CA Total Généré", f"{df_livreurs['CA_Généré'].sum():,.2f} DH")
        with col3:
            st.metric("CA Moyen/Livraison", f"{df_livreurs['CA_Moyen_Livraison'].mean():.2f} DH")
        
        st.dataframe(df_livreurs, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(df_livreurs, x='Livreur', y='Livraisons',
                           color='Véhicule', title="Nombre de livraisons par livreur",
                           labels={'Livreur': 'Livreur', 'Livraisons': 'Nombre de Livraisons'})
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(df_livreurs, x='Expérience', y='CA_Généré',
                                   size='Livraisons', color='Véhicule',
                                   hover_data=['Livreur', 'Zone'],
                                   title="Relation Expérience vs Performance",
                                   labels={'Expérience': 'Expérience (années)', 'CA_Généré': 'CA Généré (DH)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Aucune donnée de livreur disponible.")
    
    st.markdown("---")

    # Analyse par véhicule
    st.subheader("🚗 Analyse par Type de Véhicule")
    
    vehicules_query = """
    MATCH (l:Livreur)-[:DELIVERS]->(cmd:Commande)
    WITH l.vehicule as vehicule, COUNT(cmd) as nb_livraisons, 
         SUM(cmd.prix_total) as ca_total, AVG(cmd.poids_total) as poids_moyen,
         CASE WHEN COUNT(cmd) > 0 THEN ROUND(SUM(cmd.prix_total) / COUNT(cmd), 2) ELSE 0 END as ca_moyen
    RETURN vehicule as Type_Véhicule, nb_livraisons as Total_Livraisons,
           ca_total as CA_Total, poids_moyen as Poids_Moyen_kg, ca_moyen as CA_Moyen
    ORDER BY nb_livraisons DESC
    """
    
    vehicules_data = execute_safe_query(vehicules_query, "Analyse par Véhicule")
    if vehicules_data:
        df_vehicules = pd.DataFrame(vehicules_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df_vehicules, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(df_vehicules, values='Total_Livraisons', names='Type_Véhicule',
                           title="Répartition des livraisons par type de véhicule")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Aucune donnée de véhicule disponible.")

# ==============================================================================
# SECTION : ANALYSE PRODUITS
# ==============================================================================
elif page == "📦 Analyse Produits":
    st.header("📦 Analyse des Produits")
    
    st.markdown("---")

    # Produits populaires
    st.subheader("🏆 Produits les Plus Populaires")
    
    produits_query = """
    MATCH (cmd:Commande)-[c:CONTAINS]->(p:Produit)
    WITH p, SUM(c.quantite) as quantite_totale, COUNT(DISTINCT cmd) as nb_commandes,
         ROUND(p.prix * SUM(c.quantite), 2) as ca_produit
    RETURN p.nom as Produit, p.categorie as Catégorie, quantite_totale as Quantité_Vendue,
           nb_commandes as Nb_Commandes, p.prix as Prix_Unitaire, ca_produit as CA_Produit
    ORDER BY quantite_totale DESC
    """
    
    produits_data = execute_safe_query(produits_query, "Produits Populaires")
    if produits_data:
        df_produits = pd.DataFrame(produits_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Produits Différents", len(df_produits))
        with col2:
            st.metric("Quantité Totale Vendue", df_produits['Quantité_Vendue'].sum())
        with col3:
            st.metric("CA Total Produits", f"{df_produits['CA_Produit'].sum():,.2f} DH")
        
        st.dataframe(df_produits, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(df_produits.head(10), x='Produit', y='Quantité_Vendue',
                           color='Catégorie', title="Top 10 Produits par Quantité",
                           labels={'Produit': 'Produit', 'Quantité_Vendue': 'Quantité Vendue'})
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            category_stats = df_produits.groupby('Catégorie').agg({
                'Quantité_Vendue': 'sum',
                'CA_Produit': 'sum'
            }).reset_index()
            
            fig_pie = px.pie(category_stats, values='CA_Produit', names='Catégorie',
                           title="Répartition du CA par catégorie")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Aucune donnée de produit disponible.")
    
    st.markdown("---")

    # Analyse des stocks
    st.subheader("📊 Analyse des Stocks par Entrepôt")
    
    stocks_query = """
    MATCH (p:Produit)-[s:STOCKED_IN]->(e:Entrepôt)
    WITH e, p.categorie as categorie, SUM(s.quantite) as stock_total, 
         COUNT(DISTINCT p) as nb_produits
    RETURN e.nom as Entrepôt, categorie as Catégorie, 
           stock_total as Stock_Total, nb_produits as Nb_Produits
    ORDER BY e.nom, stock_total DESC
    """
    
    stocks_data = execute_safe_query(stocks_query, "Analyse des Stocks")
    if stocks_data:
        df_stocks = pd.DataFrame(stocks_data)
        
        st.dataframe(df_stocks, use_container_width=True)
        
        fig_sunburst = px.sunburst(df_stocks, path=['Entrepôt', 'Catégorie'], 
                                 values='Stock_Total',
                                 title="Répartition des stocks par entrepôt et catégorie",
                                 labels={'Stock_Total': 'Stock Total'})
        st.plotly_chart(fig_sunburst, use_container_width=True)
    else:
        st.info("Aucune donnée de stock disponible.")

# ==============================================================================
# SECTION : GESTION CLIENTS
# ==============================================================================
elif page == "👤 Gestion Clients":
    st.header("👥 Gestion des Clients")
    
    st.markdown("---")

    # Analyse des clients
    st.subheader("🎯 Analyse de la Clientèle")
    
    clients_query = """
    MATCH (c:Client)-[:ORDERED]->(cmd:Commande)
    WITH c, COUNT(cmd) as nb_commandes, SUM(cmd.prix_total) as ca_total, 
         CASE WHEN COUNT(cmd) > 0 THEN AVG(cmd.prix_total) ELSE 0 END as panier_moyen
    RETURN c.nom as Client, c.type as Type_Client, nb_commandes as Nb_Commandes,
           ROUND(ca_total, 2) as CA_Total, ROUND(panier_moyen, 2) as Panier_Moyen,
           CASE 
             WHEN nb_commandes >= 3 THEN 'VIP'
             WHEN nb_commandes >= 2 THEN 'Régulier'
             ELSE 'Occasionnel'
           END as Statut_Client
    ORDER BY nb_commandes DESC, ca_total DESC
    """
    
    clients_data = execute_safe_query(clients_query, "Analyse des Clients")
    if clients_data:
        df_clients = pd.DataFrame(clients_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clients", len(df_clients))
        with col2:
            vip_clients = len(df_clients[df_clients['Statut_Client'] == 'VIP'])
            st.metric("Clients VIP", vip_clients)
        with col3:
            avg_orders = df_clients['Nb_Commandes'].mean()
            st.metric("Commandes Moy./Client", f"{avg_orders:.1f}")
        with col4:
            avg_basket = df_clients['Panier_Moyen'].mean()
            st.metric("Panier Moyen", f"{avg_basket:.2f} DH")
        
        def color_status(val):
            if val == 'VIP':
                return 'background-color: #fffde7; color: #ffc107' # Jaune clair
            elif val == 'Régulier':
                return 'background-color: #e8f5e8; color: #4caf50' # Vert clair
            else:
                return 'background-color: #e3f2fd; color: #2196f3' # Bleu clair
        
        styled_df = df_clients.style.applymap(color_status, subset=['Statut_Client'])
        st.dataframe(styled_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_stats = df_clients.groupby('Statut_Client').size().reset_index(name='Nombre')
            fig_pie = px.pie(status_stats, values='Nombre', names='Statut_Client',
                           title="Répartition des clients par statut")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(df_clients, x='Nb_Commandes', y='CA_Total',
                                   size='Panier_Moyen', color='Type_Client',
                                   hover_data=['Client', 'Statut_Client'],
                                   title="Relation Nb Commandes vs CA",
                                   labels={'Nb_Commandes': 'Nombre de Commandes', 'CA_Total': 'CA Total (DH)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Aucune donnée client disponible.")

# ==============================================================================
# SECTION : REPORTING EXÉCUTIF
# ==============================================================================
elif page == "📊 Reporting Exécutif":
    st.header("📈 Reporting Exécutif Global")

    st.markdown("---")

    # KPIs globaux de l'agence (répétition pour le reporting)
    st.subheader("📊 Indicateurs Clés de Performance (KPIs)")
    kpi_query_report = """
    MATCH (cmd:Commande)
    WITH COUNT(cmd) as total_commandes, SUM(cmd.prix_total) as ca_total, 
         AVG(cmd.prix_total) as panier_moyen, SUM(cmd.poids_total) as poids_total
    MATCH (l:Livreur)
    WITH total_commandes, ca_total, panier_moyen, poids_total, COUNT(l) as nb_livreurs
    MATCH (c:Client)
    WITH total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, COUNT(c) as nb_clients
    MATCH (z:Zone)
    WITH total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, nb_clients, COUNT(z) as nb_zones
    MATCH (e:Entrepôt)
    WITH total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, nb_clients, nb_zones, COUNT(e) as nb_entrepots
    RETURN total_commandes, ca_total, panier_moyen, poids_total, nb_livreurs, nb_clients, nb_zones, nb_entrepots
    """

    kpi_data_report = execute_safe_query(kpi_query_report, "KPIs Report")

    if kpi_data_report:
        kpi_r = kpi_data_report[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Commandes", kpi_r.get('total_commandes', 0))
            st.metric("Nombre Clients", kpi_r.get('nb_clients', 0))

        with col2:
            st.metric("CA Total", f"{kpi_r.get('ca_total', 0):,.2f} DH")
            st.metric("Nombre Livreurs", kpi_r.get('nb_livreurs', 0))

        with col3:
            st.metric("Panier Moyen", f"{kpi_r.get('panier_moyen', 0):,.2f} DH")
            st.metric("Nombre Zones", kpi_r.get('nb_zones', 0))

        with col4:
            st.metric("Poids Total", f"{kpi_r.get('poids_total', 0):,.2f} kg")
            st.metric("Nombre Entrepôts", kpi_r.get('nb_entrepots', 0))
    else:
        st.info("Aucune donnée KPI disponible pour le reporting.")

    st.markdown("---")

    # Performance par entrepôt
    st.subheader("🏭 Performance par Entrepôt")

    entrepot_perf_query = """
    MATCH (e:Entrepôt)<-[s:STOCKED_IN]-(p:Produit)<-[c:CONTAINS]-(cmd:Commande)
    WITH e, COUNT(DISTINCT cmd) as nb_commandes_expediees, SUM(c.quantite * p.prix) as ca_expedie,
         SUM(c.quantite * p.poids) as poids_total_expedie
    RETURN e.nom as Entrepôt, nb_commandes_expediees as Commandes_Expédiées,
           ROUND(ca_expedie, 2) as CA_Expédié, ROUND(poids_total_expedie, 2) as Poids_Expédié_kg
    ORDER BY CA_Expédié DESC
    """

    entrepot_perf_data = execute_safe_query(entrepot_perf_query, "Performance Entrepôts")
    if entrepot_perf_data:
        df_entrepot_perf = pd.DataFrame(entrepot_perf_data)
        st.dataframe(df_entrepot_perf, use_container_width=True)

        fig_entrepot_bar = px.bar(df_entrepot_perf, x='Entrepôt', y='CA_Expédié',
                                  color='Commandes_Expédiées',
                                  title="Chiffre d'Affaires Expédié par Entrepôt",
                                  labels={'Entrepôt': 'Entrepôt', 'CA_Expédié': 'CA Expédié (DH)', 'Commandes_Expédiées': 'Commandes Expédiées'})
        st.plotly_chart(fig_entrepot_bar, use_container_width=True)
    else:
        st.info("Aucune donnée de performance d'entrepôt disponible.")

    st.markdown("---")

    # Tendances logistiques (évolution des statuts de commande)
    st.subheader("📊 Tendances Logistiques : Évolution des Statuts de Commande")

    tendances_statut_query = """
    MATCH (cmd:Commande)
    RETURN cmd.date_commande as date, cmd.statut as statut, COUNT(cmd) as nb_commandes
    ORDER BY date, statut
    """

    tendances_statut_data = execute_safe_query(tendances_statut_query, "Tendances Statut")
    if tendances_statut_data:
        df_tendances_statut = pd.DataFrame(tendances_statut_data)
        df_tendances_statut['date'] = pd.to_datetime(df_tendances_statut['date']) # Convertir en datetime
        fig_tendances_statut = px.line(df_tendances_statut, x='date', y='nb_commandes', color='statut',
                                       title="Évolution du Nombre de Commandes par Statut",
                                       labels={'date': 'Date', 'nb_commandes': 'Nombre de Commandes', 'statut': 'Statut'})
        st.plotly_chart(fig_tendances_statut, use_container_width=True)
    else:
        st.info("Aucune donnée de tendance de statut disponible.")

# ==============================================================================
# SECTION : ADMINISTRATION
# ==============================================================================
elif page == "🔧 Administration":
    st.header("⚙️ Outils d'Administration et de Maintenance")

    st.markdown("---")

    st.subheader("🔍 Informations sur la Base de Données")
    db_info_query = """
    CALL db.labels() YIELD label
    RETURN label, size((:label)) as count
    UNION ALL
    CALL db.relationshipTypes() YIELD relationshipType
    RETURN relationshipType, size(()-[relationshipType]->()) as count
    """
    db_info_data = execute_safe_query(db_info_query, "DB Info")
    if db_info_data:
        df_db_info = pd.DataFrame(db_info_data)
        st.dataframe(df_db_info, use_container_width=True)
    else:
        st.info("Impossible de récupérer les informations de la base de données.")

    st.markdown("---")

    st.subheader("🧹 Nettoyage et Optimisation")
    st.warning("🚨 Attention: Ces opérations sont irréversibles et peuvent supprimer toutes vos données. Utilisez avec prudence.")

    if st.button("🗑️ Supprimer TOUTES les données (CLEANUP)"):
        confirm_delete = st.checkbox("Je suis sûr de vouloir supprimer toutes les données.")
        if confirm_delete:
            cleanup_query = """
            MATCH (n) DETACH DELETE n
            """
            try:
                conn.execute_query(cleanup_query)
                st.success("✅ Toutes les données ont été supprimées de la base de données.")
                st.cache_resource.clear() # Efface le cache pour recharger la connexion si nécessaire
                st.experimental_rerun() # Recharge l'application pour refléter les changements
            except Exception as e:
                st.error(f"❌ Erreur lors du nettoyage : {e}")
        else:
            st.info("Veuillez cocher la case de confirmation pour activer la suppression.")

    st.markdown("---")

    st.subheader("➕ Génération de Données de Test")
    st.info("Permet de recréer un jeu de données de base pour les démonstrations ou le développement.")

    if st.button("✨ Générer des Données de Test"):
        try:
            # Suppression préalable pour éviter les doublons si déjà des données
            conn.execute_query("MATCH (n) DETACH DELETE n")

            # Création de nœuds
            conn.execute_query("""
            CREATE (c1:Client {id: 'C001', nom: 'Client A', type: 'Particulier'})
            CREATE (c2:Client {id: 'C002', nom: 'Client B', type: 'Entreprise'})
            CREATE (c3:Client {id: 'C003', nom: 'Client C', type: 'Particulier'})
            CREATE (c4:Client {id: 'C004', nom: 'Client D', type: 'Entreprise'})
            CREATE (c5:Client {id: 'C005', nom: 'Client E', type: 'Particulier'})

            CREATE (l1:Livreur {id: 'L001', nom: 'Dupont', vehicule: 'Moto', experience: 3})
            CREATE (l2:Livreur {id: 'L002', nom: 'Martin', vehicule: 'Camionnette', experience: 5})
            CREATE (l3:Livreur {id: 'L003', nom: 'Bernard', vehicule: 'Vélo', experience: 1})
            CREATE (l4:Livreur {id: 'L004', nom: 'Dubois', vehicule: 'Camionnette', experience: 7})

            CREATE (e1:Entrepôt {id: 'E001', nom: 'Entrepôt Nord', capacite: 1000})
            CREATE (e2:Entrepôt {id: 'E002', nom: 'Entrepôt Sud', capacite: 1500})
            CREATE (e3:Entrepôt {id: 'E003', nom: 'Entrepôt Ouest', capacite: 800})

            CREATE (p1:Produit {id: 'P001', nom: 'Ordinateur Portable', categorie: 'Électronique', prix: 1200, poids: 2.5})
            CREATE (p2:Produit {id: 'P002', nom: 'Livre "Neo4j"', categorie: 'Livre', prix: 30, poids: 0.5})
            CREATE (p3:Produit {id: 'P003', nom: 'Cafetière', categorie: 'Électroménager', prix: 80, poids: 1.2})
            CREATE (p4:Produit {id: 'P004', nom: 'Smartphone', categorie: 'Électronique', prix: 800, poids: 0.3})
            CREATE (p5:Produit {id: 'P005', nom: 'Casque Audio', categorie: 'Électronique', prix: 150, poids: 0.2})

            CREATE (z1:Zone {id: 'Z001', nom: 'Centre Ville', densite_population: 'Élevée'})
            CREATE (z2:Zone {id: 'Z002', nom: 'Banlieue Ouest', densite_population: 'Moyenne'})
            CREATE (z3:Zone {id: 'Z003', nom: 'Zone Industrielle', densite_population: 'Faible'})
            CREATE (z4:Zone {id: 'Z004', nom: 'Quartier Résidentiel', densite_population: 'Moyenne'})
            """)

            # Création de relations
            conn.execute_query("""
            MATCH (c1:Client {id: 'C001'}), (z1:Zone {id: 'Z001'}) CREATE (c1)-[:LOCATED_IN]->(z1)
            MATCH (c2:Client {id: 'C002'}), (z2:Zone {id: 'Z002'}) CREATE (c2)-[:LOCATED_IN]->(z2)
            MATCH (c3:Client {id: 'C003'}), (z1:Zone {id: 'Z001'}) CREATE (c3)-[:LOCATED_IN]->(z1)
            MATCH (c4:Client {id: 'C004'}), (z3:Zone {id: 'Z003'}) CREATE (c4)-[:LOCATED_IN]->(z3)
            MATCH (c5:Client {id: 'C005'}), (z4:Zone {id: 'Z004'}) CREATE (c5)-[:LOCATED_IN]->(z4)

            MATCH (l1:Livreur {id: 'L001'}), (z1:Zone {id: 'Z001'}) CREATE (l1)-[:ASSIGNED_TO]->(z1)
            MATCH (l2:Livreur {id: 'L002'}), (z2:Zone {id: 'Z002'}) CREATE (l2)-[:ASSIGNED_TO]->(z2)
            MATCH (l3:Livreur {id: 'L003'}), (z1:Zone {id: 'Z001'}) CREATE (l3)-[:ASSIGNED_TO]->(z1)
            MATCH (l4:Livreur {id: 'L004'}), (z3:Zone {id: 'Z003'}) CREATE (l4)-[:ASSIGNED_TO]->(z3)

            MATCH (p1:Produit {id: 'P001'}), (e1:Entrepôt {id: 'E001'}) CREATE (p1)-[:STOCKED_IN {quantite: 50}]->(e1)
            MATCH (p2:Produit {id: 'P002'}), (e2:Entrepôt {id: 'E002'}) CREATE (p2)-[:STOCKED_IN {quantite: 200}]->(e2)
            MATCH (p3:Produit {id: 'P003'}), (e1:Entrepôt {id: 'E001'}) CREATE (p3)-[:STOCKED_IN {quantite: 100}]->(e1)
            MATCH (p4:Produit {id: 'P004'}), (e3:Entrepôt {id: 'E003'}) CREATE (p4)-[:STOCKED_IN {quantite: 75}]->(e3)
            MATCH (p5:Produit {id: 'P005'}), (e2:Entrepôt {id: 'E002'}) CREATE (p5)-[:STOCKED_IN {quantite: 120}]->(e2)
            """)

            # Création de commandes et relations complexes
            conn.execute_query("""
            MATCH (c1:Client {id: 'C001'}), (l1:Livreur {id: 'L001'}), (p1:Produit {id: 'P001'}), (p2:Produit {id: 'P002'})
            CREATE (cmd1:Commande {id: 'CMD001', date_commande: date('2023-01-15'), prix_total: 1230, poids_total: 3.0, statut: 'Livré'})
            CREATE (c1)-[:ORDERED]->(cmd1)
            CREATE (l1)-[:DELIVERS]->(cmd1)
            CREATE (cmd1)-[:CONTAINS {quantite: 1}]->(p1)
            CREATE (cmd1)-[:CONTAINS {quantite: 1}]->(p2)

            MATCH (c2:Client {id: 'C002'}), (l2:Livreur {id: 'L002'}), (p3:Produit {id: 'P003'})
            CREATE (cmd2:Commande {id: 'CMD002', date_commande: date('2023-01-16'), prix_total: 80, poids_total: 1.2, statut: 'En cours'})
            CREATE (c2)-[:ORDERED]->(cmd2)
            CREATE (l2)-[:DELIVERS]->(cmd2)
            CREATE (cmd2)-[:CONTAINS {quantite: 1}]->(p3)

            MATCH (c3:Client {id: 'C003'}), (l1:Livreur {id: 'L001'}), (p4:Produit {id: 'P004'})
            CREATE (cmd3:Commande {id: 'CMD003', date_commande: date('2023-01-17'), prix_total: 800, poids_total: 0.3, statut: 'Livré'})
            CREATE (c3)-[:ORDERED]->(cmd3)
            CREATE (l1)-[:DELIVERS]->(cmd3)
            CREATE (cmd3)-[:CONTAINS {quantite: 1}]->(p4)

            MATCH (c4:Client {id: 'C004'}), (l4:Livreur {id: 'L004'}), (p1:Produit {id: 'P001'}), (p5:Produit {id: 'P005'})
            CREATE (cmd4:Commande {id: 'CMD004', date_commande: date('2023-01-18'), prix_total: 1350, poids_total: 2.7, statut: 'En attente'})
            CREATE (c4)-[:ORDERED]->(cmd4)
            CREATE (l4)-[:DELIVERS]->(cmd4)
            CREATE (cmd4)-[:CONTAINS {quantite: 1}]->(p1)
            CREATE (cmd4)-[:CONTAINS {quantite: 1}]->(p5)

            MATCH (c5:Client {id: 'C005'}), (l2:Livreur {id: 'L002'}), (p2:Produit {id: 'P002'})
            CREATE (cmd5:Commande {id: 'CMD005', date_commande: date('2023-01-19'), prix_total: 60, poids_total: 1.0, statut: 'Livré'})
            CREATE (c5)-[:ORDERED]->(cmd5)
            CREATE (l2)-[:DELIVERS]->(cmd5)
            CREATE (cmd5)-[:CONTAINS {quantite: 2}]->(p2)
            """)

            # Création de trajets
            conn.execute_query("""
            MATCH (e1:Entrepôt {id: 'E001'}), (z1:Zone {id: 'Z001'})
            CREATE (t1:Trajet {id: 'TRJ001', origine: 'E001', distance: 15, duree: 30, cout: 150})
            CREATE (t1)-[:PASSED_BY]->(z1)

            MATCH (e2:Entrepôt {id: 'E002'}), (z2:Zone {id: 'Z002'})
            CREATE (t2:Trajet {id: 'TRJ002', origine: 'E002', distance: 25, duree: 45, cout: 200})
            CREATE (t2)-[:PASSED_BY]->(z2)

            MATCH (e1:Entrepôt {id: 'E001'}), (z4:Zone {id: 'Z004'})
            CREATE (t3:Trajet {id: 'TRJ003', origine: 'E001', distance: 20, duree: 40, cout: 180})
            CREATE (t3)-[:PASSED_BY]->(z4)

            MATCH (e3:Entrepôt {id: 'E003'}), (z3:Zone {id: 'Z003'})
            CREATE (t4:Trajet {id: 'TRJ004', origine: 'E003', distance: 10, duree: 20, cout: 90})
            CREATE (t4)-[:PASSED_BY]->(z3)
            """)

            st.success("✅ Données de test générées avec succès. Actualisation de la page...")
            st.cache_resource.clear() # Efface le cache pour recharger la connexion si nécessaire
            st.experimental_rerun() # Recharge l'application pour refléter les nouvelles données
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération des données de test : {e}")

# Note: La fermeture de la connexion Neo4j est gérée par st.cache_resource.
# Elle sera fermée automatiquement lorsque l'application Streamlit s'arrêtera.
