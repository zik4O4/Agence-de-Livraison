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
    page_icon="icon.png",
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

def create_entity(entity_type, properties):
    """Crée un nouveau nœud avec les propriétés données"""
    query = f"""
    MERGE (n:{entity_type} {{id: $id}})
    SET n += $properties
    """
    try:
        params = {"id": properties.pop("id"), "properties": properties}
        conn.execute_query(query, params)
        st.success(f"{entity_type} {properties['nom']} créé avec succès!")
    except Exception as e:
        st.error(f"Erreur création {entity_type}: {str(e)}")

def create_relationship(from_type, from_id, to_type, to_id, rel_type):
    """Établit une relation entre deux nœuds"""
    query = f"""
    MATCH (a:{from_type} {{id: $from_id}}), (b:{to_type} {{id: $to_id}})
    MERGE (a)-[r:{rel_type}]->(b)
    RETURN type(r) as relation_type
    """
    try:
        result = conn.execute_query(query, {"from_id": from_id, "to_id": to_id})
        if result:
            st.success(f"Relation {result[0]['relation_type']} créée!")
        else:
            st.warning("Un des nœuds n'existe pas")
    except Exception as e:
        st.error(f"Erreur création relation: {str(e)}")

def process_csv_import(file):
    """Traite un fichier CSV d'import"""
    try:
        df = pd.read_csv(file)
        required_cols = {"type_entite", "id", "nom"}
        
        if not required_cols.issubset(df.columns):
            st.error(f"Colonnes obligatoires manquantes. Requises: {', '.join(required_cols)}")
            return
            
        for _, row in df.iterrows():
            row_data = row.dropna().to_dict()
            create_entity(row['type_entite'], row_data)
            
        st.success(f"Import réussi: {len(df)} entités ajoutées")
    except Exception as e:
        st.error(f"Erreur d'import: {str(e)}")

def display_entities(entity_type):
    """Affiche les entités existantes"""
    query = f"MATCH (n:{entity_type}) RETURN n LIMIT 100"
    result = conn.execute_query(query)
    
    if not result:
        st.info(f"Aucun(e) {entity_type} trouvé(e)")
        return
        
    data = []
    for record in result:
        node = record["n"]
        data.append({
            "id": node.get("id", ""),
            "nom": node.get("nom", ""),
            **{k: v for k, v in node.items() if k not in ["id", "nom"]}
        })
    
    st.dataframe(pd.DataFrame(data), use_container_width=True)

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
        st.error(f"Erreur de connexion à Neo4j: {e}. Veuillez vérifier l'URI, l'utilisateur et le mot de passe.")
        st.stop() # Arrête l'exécution de l'application si la connexion échoue
        return None

# Header principal de l'application

st.markdown("""
<div class="main-header">
    <h1>
    Agence de Livraison - Dashboard Neo4j</h1>
    <p>Système de gestion et d'optimisation des flux logistiques</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour la navigation


from streamlit_option_menu import option_menu
# Sidebar navigation with icons
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  # Required
        options=[
            "Tableau de Bord", 
            "Analyse des Zones",
            "Optimisation Trajets",
            "Gestion Livreurs",
            "Analyse Produits",
            "Gestion Clients",
            "Reporting Exécutif",
            "Importer Données",
            "Administration"
        ],
        icons=[
            "speedometer2", 
            "map",
            "geo-alt",
            "people", 
            "box-seam",
            "person", 
            "graph-up",
            "upload", 
            "gear"
        ],
        menu_icon="menu-button",  # Optional
        default_index=0,  # Optional
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "whith", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#007bff", "color": "white"},  # Highlight selected link
        }
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
if selected == "Tableau de Bord":
    st.header("Vue d'ensemble de l'agence")
    
    st.markdown("---")

    # KPIs principaux
    st.subheader("Indicateurs Clés de Performance")
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
        kpi = kpi_data[0] # kpi est un dictionnaire

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Commandes", kpi.get('total_commandes', 0))
            st.metric("Nombre Clients", kpi.get('nb_clients', 0))

        with col2:
            ca_total_value = kpi.get('ca_total')
            if ca_total_value is None:
                st.metric("CA Total", "N/A")
            else:
                st.metric("CA Total", f"{ca_total_value:,.2f} DH")

            st.metric("Nombre Livreurs", kpi.get('nb_livreurs', 0))

        with col3:
            panier_moyen_value = kpi.get('panier_moyen')
            if panier_moyen_value is None:
                st.metric("Panier Moyen", "N/A")
            else:
                st.metric("Panier Moyen", f"{panier_moyen_value:,.2f} DH")

            st.metric("Nombre Zones", kpi.get('nb_zones', 0))

        with col4:
            poids_total_value = kpi.get('poids_total')
            if poids_total_value is None:
                st.metric("Poids Total", "N/A")
            else:
                st.metric("Poids Total", f"{poids_total_value:,.2f} kg")

            st.metric("Nombre Entrepôts", kpi.get('nb_entrepots', 0))
    else:
        st.warning("Aucune donnée KPI disponible. Veuillez générer des données de test si ce n'est pas déjà fait.")



    st.markdown("---")

    # Graphiques de synthèse
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Évolution des Commandes")
        evolution_query = """
        MATCH (cmd:Commande)
        RETURN cmd.date_commande as date, COUNT(cmd) as nb_commandes, 
             SUM(cmd.prix_total) as ca_jour
        ORDER BY date
        """
        
        evolution_data = execute_safe_query(evolution_query, "Évolution des Commandes")
        if evolution_data:
            df_evolution = pd.DataFrame(evolution_data)
            df_evolution['date'] = df_evolution['date'].apply(lambda x: pd.Timestamp(x.to_native()).date())
            fig_evolution = px.line(df_evolution, x='date', y='nb_commandes', 
                                  title="Nombre de commandes par jour",
                                  labels={'date': 'Date', 'nb_commandes': 'Nombre de Commandes'})
            st.plotly_chart(fig_evolution, use_container_width=True)
        else:
            st.info("Aucune donnée d'évolution des commandes disponible.")
    
    with col2:
        st.subheader("Top Zones par CA")
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
    st.subheader("Statut des Commandes")
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
elif selected == "Analyse des Zones":
    st.header("Analyse des Zones de Livraison")
    
    st.markdown("---")

    # Zones à forte densité
    st.subheader("Zones à Forte Densité de Commandes")
    
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
    st.subheader("Détection des Goulets d'Étranglement")
    
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
elif selected == "Optimisation Trajets":
    st.header("Optimisation des Itinéraires")
    
    st.markdown("---")

    # Analyse des trajets coûteux
    st.subheader("Analyse des Coûts de Transport")
    
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
    st.subheader("Suggestions d'Optimisation")
    
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
elif selected == "Gestion Livreurs":
    st.header("Gestion des Livreurs")
    
    st.markdown("---")

    # Performance des livreurs
    st.subheader("Performance des Livreurs")
    
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
    st.subheader("Analyse par Type de Véhicule")
    
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
elif selected == "Analyse Produits":
    st.header("Analyse des Produits")
    
    st.markdown("---")

    # Produits populaires
    st.subheader("Produits les Plus Populaires")
    
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
    st.subheader("Analyse des Stocks par Entrepôt")
    
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
elif selected == "Gestion Clients":
    st.header("Gestion des Clients")
    
    st.markdown("---")

    # Analyse des clients
    st.subheader("Analyse de la Clientèle")
    
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

elif selected == "Reporting Exécutif":
    st.header("Reporting Exécutif")

    # KPIs principaux pour le reporting
    st.subheader("Vue d'ensemble des Performances")
    kpi_reporting_query = """
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

    kpi_reporting_data = execute_safe_query(kpi_reporting_query, "Reporting KPIs")

    if kpi_reporting_data:
        kpi_r = kpi_reporting_data[0] # kpi_r est le dictionnaire de résultats

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Commandes", kpi_r.get('total_commandes', 0))
            st.metric("Nombre Clients", kpi_r.get('nb_clients', 0))

        with col2:
            ca_total_r_value = kpi_r.get('ca_total')
            if ca_total_r_value is None:
                st.metric("CA Total", "N/A")
            else:
                st.metric("CA Total", f"{ca_total_r_value:,.2f} DH")

            st.metric("Nombre Livreurs", kpi_r.get('nb_livreurs', 0))

        with col3:
            panier_moyen_r_value = kpi_r.get('panier_moyen')
            if panier_moyen_r_value is None:
                st.metric("Panier Moyen", "N/A")
            else:
                st.metric("Panier Moyen", f"{panier_moyen_r_value:,.2f} DH")

            st.metric("Nombre Zones", kpi_r.get('nb_zones', 0))

        with col4:
            poids_total_r_value = kpi_r.get('poids_total')
            if poids_total_r_value is None:
                st.metric("Poids Total", "N/A")
            else:
                st.metric("Poids Total", f"{poids_total_r_value:,.2f} kg")

            st.metric("Nombre Entrepôts", kpi_r.get('nb_entrepots', 0))
    else:
        st.warning("Aucune donnée KPI disponible pour le reporting exécutif. Veuillez générer des données de test si ce n'est pas déjà fait.")

    # Graphiques de tendance des commandes et CA
    st.subheader("Tendance des Commandes et du Chiffre d'Affaires")
    reporting_evolution_query = """
    MATCH (cmd:Commande)
    WITH cmd.date_commande as date, COUNT(cmd) as nb_commandes,
         SUM(cmd.prix_total) as ca_jour
    RETURN date, nb_commandes, ca_jour
    ORDER BY date
    """
    reporting_evolution_data = execute_safe_query(reporting_evolution_query, "Reporting Évolution")


    if reporting_evolution_data:
        df_reporting_evolution = pd.DataFrame(reporting_evolution_data)
        df_reporting_evolution['date'] = df_reporting_evolution['date'].apply(lambda d: datetime(d.year, d.month, d.day)) # Convertir en datetime

        fig_reporting_evolution = make_subplots(specs=[[{"secondary_y": True}]])

        fig_reporting_evolution.add_trace(
            go.Scatter(x=df_reporting_evolution['date'], y=df_reporting_evolution['nb_commandes'], name="Nombre de Commandes"),
            secondary_y=False,
        )

        fig_reporting_evolution.add_trace(
            go.Scatter(x=df_reporting_evolution['date'], y=df_reporting_evolution['ca_jour'], name="Chiffre d'Affaires (DH)"),
            secondary_y=True,
        )

        fig_reporting_evolution.update_layout(title_text="Évolution Quotidienne des Commandes et du Chiffre d'Affaires")
        fig_reporting_evolution.update_xaxes(title_text="Date")
        fig_reporting_evolution.update_yaxes(title_text="Nombre de Commandes", secondary_y=False)
        fig_reporting_evolution.update_yaxes(title_text="Chiffre d'Affaires (DH)", secondary_y=True)

        st.plotly_chart(fig_reporting_evolution, use_container_width=True)
    else:
        st.info("Pas de données d'évolution des commandes disponibles pour le reporting.")

    # Répartition des commandes par statut
    st.subheader("Répartition des Commandes par Statut")
    reporting_statut_query = """
    MATCH (cmd:Commande)
    WITH cmd.statut as statut, COUNT(cmd) as nb_commandes,
         SUM(cmd.prix_total) as ca_statut
    RETURN statut, nb_commandes, ca_statut
    ORDER BY nb_commandes DESC
    """
    reporting_statut_data = execute_safe_query(reporting_statut_query, "Reporting Statut")

    if reporting_statut_data:
        df_reporting_statut = pd.DataFrame(reporting_statut_data)

        col1, col2 = st.columns(2)
        with col1:
            fig_pie_statut = px.pie(df_reporting_statut, values='nb_commandes', names='statut',
                                    title="Répartition des Commandes par Statut")
            st.plotly_chart(fig_pie_statut, use_container_width=True)

        with col2:
            fig_bar_ca_statut = px.bar(df_reporting_statut, x='statut', y='ca_statut',
                                       title="Chiffre d'Affaires par Statut de Commande")
            st.plotly_chart(fig_bar_ca_statut, use_container_width=True)
    else:
        st.info("Pas de données de statut de commande disponibles pour le reporting.")

    # Top 5 des livreurs par CA généré
    st.subheader("Top 5 Livreurs par CA Généré")
    reporting_top_livreurs_query = """
    MATCH (l:Livreur)-[:DELIVERS]->(cmd:Commande)
    RETURN l.nom as Livreur, SUM(cmd.prix_total) as CA_Généré
    ORDER BY CA_Généré DESC
    LIMIT 5
    """
    reporting_top_livreurs_data = execute_safe_query(reporting_top_livreurs_query, "Reporting Top Livreurs")

    if reporting_top_livreurs_data:
        df_reporting_top_livreurs = pd.DataFrame(reporting_top_livreurs_data)
        fig_top_livreurs = px.bar(df_reporting_top_livreurs, x='Livreur', y='CA_Généré',
                                  title="Top 5 Livreurs par Chiffre d'Affaires Généré")
        st.plotly_chart(fig_top_livreurs, use_container_width=True)
    else:
        st.info("Pas de données de top livreurs disponibles pour le reporting.")


# ==================== IMPORT DE DONNEES ==================== 
elif selected == "Importer Données":
    st.header("Interface d'Import Métier")
    
    # Section 1: Ajout manuel
    with st.expander("Ajout Manuel", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form("Nouvelle Entité"):
                st.subheader("Créer une Entité")
                entity_type = st.selectbox("Type d'entité", 
                    ["Client", "Produit", "Livreur", "Entrepôt", "Zone"])
                
                entity_id = st.text_input("ID Unique*")
                entity_name = st.text_input("Nom*")
                
                # Champs spécifiques par type
                if entity_type == "Client":
                    client_type = st.selectbox("Type Client", ["Particulier", "Entreprise"])
                elif entity_type == "Produit":
                    product_cat = st.text_input("Catégorie*")
                    product_price = st.number_input("Prix*", min_value=0.0)
                    product_weight = st.number_input("Poids (kg)", min_value=0.0)
                
                if st.form_submit_button(f"Créer {entity_type}"):
                    if not entity_id or not entity_name:
                        st.warning("Les champs obligatoires (*) doivent être remplis")
                    else:
                        create_entity(entity_type, {
                            "id": entity_id,
                            "nom": entity_name,
                            **({"type": client_type} if entity_type == "Client" else {}),
                            **({"categorie": product_cat, "prix": product_price, 
                               "poids": product_weight} if entity_type == "Produit" else {})
                        })

        with col2:
            with st.form("Nouvelle Relation"):
                st.subheader("Créer une Relation")
                rel_types = ["LOCATED_IN", "STOCKED_IN", "DELIVERS", "ORDERED", "ASSIGNED_TO"]
                rel_type = st.selectbox("Type de relation*", rel_types)
                
                st.markdown("**Nœud Source**")
                from_type = st.selectbox("Type source", ["Client", "Produit", "Livreur"])
                from_id = st.text_input("ID source*")
                
                st.markdown("**Nœud Cible**")
                to_type = st.selectbox("Type cible", ["Zone", "Entrepôt", "Commande"])
                to_id = st.text_input("ID cible*")
                
                if st.form_submit_button("Établir Relation"):
                    if not from_id or not to_id:
                        st.warning("IDs source et cible obligatoires")
                    else:
                        create_relationship(from_type, from_id, to_type, to_id, rel_type)
    
    # Section 2: Import CSV
    with st.expander("Import par Fichier CSV", expanded=True):
        st.info("Format requis: Fichier CSV avec colonnes correspondant aux propriétés des nœuds")
        uploaded_file = st.file_uploader("Choissisez un fichier CSV", type="csv")
        
        if uploaded_file:
            if st.button("Lancer l'Import"):
                process_csv_import(uploaded_file)

    # Section 3: Prévisualisation données existantes
    with st.expander("Vérifier les Données Existant"):
        entity_to_check = st.selectbox("Voir tous les", 
            ["Clients", "Produits", "Livreurs", "Commandes"])
        if st.button("Afficher"):
            display_entities(entity_to_check[:-1])  # Retire le 's' final

# ==============================================================================
# SECTION : ADMINISTRATION
# ==============================================================================
elif selected == "Administration":
    st.header("Outils d'Administration et de Maintenance")

    st.markdown("---")

    st.subheader("Informations sur la Base de Données")
    db_info_query = """
    CALL db.labels() YIELD label
    MATCH (n) WHERE head(labels(n)) = label
    RETURN label AS name, count(n) as count // Renommé 'label' en 'name'
    UNION ALL
    CALL db.relationshipTypes() YIELD relationshipType
    MATCH ()-[r]->() WHERE type(r) = relationshipType
    RETURN relationshipType AS name, count(r) as count // Renommé 'relationshipType' en 'name'
    """

    db_info_data = execute_safe_query(db_info_query, "DB Info")
    if db_info_data:
        df_db_info = pd.DataFrame(db_info_data)
        st.dataframe(df_db_info, use_container_width=True)
    else:
        st.info("Impossible de récupérer les informations de la base de données.")

    st.markdown("---")

    st.subheader("Nettoyage et Optimisation")
    st.warning("Attention: Ces opérations sont irréversibles et peuvent supprimer toutes vos données. Utilisez avec prudence.")

    if st.button("Supprimer TOUTES les données (CLEANUP)"):
        cleanup_query = """
        MATCH (n) DETACH DELETE n
        """
        try:
            # Vérifiez si la connexion est établie
            if conn is not None:
                conn.execute_query(cleanup_query)
                st.success("Toutes les données ont été supprimées de la base de données.")
                st.cache_resource.clear()  # Efface le cache pour recharger la connexion si nécessaire
                st.experimental_rerun()  # Recharge l'application pour refléter les changements
            else:
                st.error("La connexion à la base de données n'est pas établie.")
        except Exception as e:
            st.error(f"Erreur lors du nettoyage : {e}")

        

    st.markdown("---")

    st.subheader("Génération de Données de Test")
    st.info("Permet de recréer un jeu de données de base pour les démonstrations ou le développement.")

    if st.button("Générer des Données de Test"):
        try:
            # Suppression des données existantes
            conn.execute_query("MATCH (n) DETACH DELETE n")

            # Création des clients (une requête par instruction)
            conn.execute_query("CREATE (c1:Client {id: 'C001', nom: 'Client A', type: 'Particulier'})")
            conn.execute_query("CREATE (c2:Client {id: 'C002', nom: 'Client B', type: 'Entreprise'})")
            conn.execute_query("CREATE (c3:Client {id: 'C003', nom: 'Client C', type: 'Particulier'})")
            conn.execute_query("CREATE (c4:Client {id: 'C004', nom: 'Client D', type: 'Entreprise'})")
            conn.execute_query("CREATE (c5:Client {id: 'C005', nom: 'Client E', type: 'Particulier'})")

            # Création des livreurs
            conn.execute_query("CREATE (l1:Livreur {id: 'L001', nom: 'Dupont', vehicule: 'Moto', experience: 3})")
            conn.execute_query("CREATE (l2:Livreur {id: 'L002', nom: 'Martin', vehicule: 'Camionnette', experience: 5})")
            conn.execute_query("CREATE (l3:Livreur {id: 'L003', nom: 'Bernard', vehicule: 'Vélo', experience: 1})")
            conn.execute_query("CREATE (l4:Livreur {id: 'L004', nom: 'Dubois', vehicule: 'Camionnette', experience: 7})")

            # Création des entrepôts
            conn.execute_query("CREATE (e1:Entrepôt {id: 'E001', nom: 'Entrepôt Nord', capacite: 1000})")
            conn.execute_query("CREATE (e2:Entrepôt {id: 'E002', nom: 'Entrepôt Sud', capacite: 1500})")
            conn.execute_query("CREATE (e3:Entrepôt {id: 'E003', nom: 'Entrepôt Ouest', capacite: 800})")

            # Création des produits
            conn.execute_query("CREATE (p1:Produit {id: 'P001', nom: 'Ordinateur Portable', categorie: 'Électronique', prix: 1200, poids: 2.5})")
            conn.execute_query("CREATE (p2:Produit {id: 'P002', nom: 'Livre \"Neo4j\"', categorie: 'Livre', prix: 30, poids: 0.5})")
            conn.execute_query("CREATE (p3:Produit {id: 'P003', nom: 'Cafetière', categorie: 'Électroménager', prix: 80, poids: 1.2})")
            conn.execute_query("CREATE (p4:Produit {id: 'P004', nom: 'Smartphone', categorie: 'Électronique', prix: 800, poids: 0.3})")
            conn.execute_query("CREATE (p5:Produit {id: 'P005', nom: 'Casque Audio', categorie: 'Électronique', prix: 150, poids: 0.2})")

            # Création des zones
            conn.execute_query("CREATE (z1:Zone {id: 'Z001', nom: 'Centre Ville', densite_population: 'Élevée'})")
            conn.execute_query("CREATE (z2:Zone {id: 'Z002', nom: 'Banlieue Ouest', densite_population: 'Moyenne'})")
            conn.execute_query("CREATE (z3:Zone {id: 'Z003', nom: 'Zone Industrielle', densite_population: 'Faible'})")
            conn.execute_query("CREATE (z4:Zone {id: 'Z004', nom: 'Quartier Résidentiel', densite_population: 'Moyenne'})")

            # Création des relations LOCATED_IN
            conn.execute_query("MATCH (c1:Client {id: 'C001'}), (z1:Zone {id: 'Z001'}) CREATE (c1)-[:LOCATED_IN]->(z1)")
            conn.execute_query("MATCH (c2:Client {id: 'C002'}), (z2:Zone {id: 'Z002'}) CREATE (c2)-[:LOCATED_IN]->(z2)")
            conn.execute_query("MATCH (c3:Client {id: 'C003'}), (z1:Zone {id: 'Z001'}) CREATE (c3)-[:LOCATED_IN]->(z1)")
            conn.execute_query("MATCH (c4:Client {id: 'C004'}), (z3:Zone {id: 'Z003'}) CREATE (c4)-[:LOCATED_IN]->(z3)")
            conn.execute_query("MATCH (c5:Client {id: 'C005'}), (z4:Zone {id: 'Z004'}) CREATE (c5)-[:LOCATED_IN]->(z4)")

            # Création des relations ASSIGNED_TO
            conn.execute_query("MATCH (l1:Livreur {id: 'L001'}), (z1:Zone {id: 'Z001'}) CREATE (l1)-[:ASSIGNED_TO]->(z1)")
            conn.execute_query("MATCH (l2:Livreur {id: 'L002'}), (z2:Zone {id: 'Z002'}) CREATE (l2)-[:ASSIGNED_TO]->(z2)")
            conn.execute_query("MATCH (l3:Livreur {id: 'L003'}), (z1:Zone {id: 'Z001'}) CREATE (l3)-[:ASSIGNED_TO]->(z1)")
            conn.execute_query("MATCH (l4:Livreur {id: 'L004'}), (z3:Zone {id: 'Z003'}) CREATE (l4)-[:ASSIGNED_TO]->(z3)")

            # Création des relations STOCKED_IN
            conn.execute_query("MATCH (p1:Produit {id: 'P001'}), (e1:Entrepôt {id: 'E001'}) CREATE (p1)-[:STOCKED_IN {quantite: 50}]->(e1)")
            conn.execute_query("MATCH (p2:Produit {id: 'P002'}), (e2:Entrepôt {id: 'E002'}) CREATE (p2)-[:STOCKED_IN {quantite: 200}]->(e2)")
            conn.execute_query("MATCH (p3:Produit {id: 'P003'}), (e1:Entrepôt {id: 'E001'}) CREATE (p3)-[:STOCKED_IN {quantite: 100}]->(e1)")
            conn.execute_query("MATCH (p4:Produit {id: 'P004'}), (e3:Entrepôt {id: 'E003'}) CREATE (p4)-[:STOCKED_IN {quantite: 75}]->(e3)")
            conn.execute_query("MATCH (p5:Produit {id: 'P005'}), (e2:Entrepôt {id: 'E002'}) CREATE (p5)-[:STOCKED_IN {quantite: 120}]->(e2)")

            # Création des commandes et leurs relations
            def create_commande(client_id, livreur_id, produits, date_cmd, statut):
                # Crée une commande avec plusieurs produits
                conn.execute_query(f"""
                MATCH (c:Client {{id: '{client_id}'}}), (l:Livreur {{id: '{livreur_id}'}})
                CREATE (cmd:Commande {{
                    id: 'CMD{client_id[-1]}', 
                    date_commande: date('{date_cmd}'), 
                    prix_total: {sum(p['prix']*p['qte'] for p in produits)}, 
                    poids_total: {sum(p['poids']*p['qte'] for p in produits)}, 
                    statut: '{statut}'
                }})
                CREATE (c)-[:ORDERED]->(cmd)
                CREATE (l)-[:DELIVERS]->(cmd)
                """)
                
                # Ajoute les produits
                for produit in produits:
                    conn.execute_query(f"""
                    MATCH (cmd:Commande {{id: 'CMD{client_id[-1]}'}}), (p:Produit {{id: '{produit['id']}'}})
                    CREATE (cmd)-[:CONTAINS {{quantite: {produit['qte']}}}]->(p)
                    """)

            # Exemple de commandes
            create_commande('C001', 'L001', [{'id':'P001', 'prix':1200, 'poids':2.5, 'qte':1}, {'id':'P002', 'prix':30, 'poids':0.5, 'qte':1}], '2023-01-15', 'Livré')
            create_commande('C002', 'L002', [{'id':'P003', 'prix':80, 'poids':1.2, 'qte':1}], '2023-01-16', 'En cours')
            create_commande('C003', 'L001', [{'id':'P004', 'prix':800, 'poids':0.3, 'qte':1}], '2023-01-17', 'Livré')
            create_commande('C004', 'L004', [{'id':'P001', 'prix':1200, 'poids':2.5, 'qte':1}, {'id':'P005', 'prix':150, 'poids':0.2, 'qte':1}], '2023-01-18', 'En attente')
            create_commande('C005', 'L002', [{'id':'P002', 'prix':30, 'poids':0.5, 'qte':2}], '2023-01-19', 'Livré')

            # Création des trajets
            conn.execute_query("MATCH (e1:Entrepôt {id: 'E001'}), (z1:Zone {id: 'Z001'}) CREATE (t1:Trajet {id: 'TRJ001', origine: 'E001', distance: 15, duree: 30, cout: 150})-[:PASSED_BY]->(z1)")
            conn.execute_query("MATCH (e2:Entrepôt {id: 'E002'}), (z2:Zone {id: 'Z002'}) CREATE (t2:Trajet {id: 'TRJ002', origine: 'E002', distance: 25, duree: 45, cout: 200})-[:PASSED_BY]->(z2)")
            conn.execute_query("MATCH (e1:Entrepôt {id: 'E001'}), (z4:Zone {id: 'Z004'}) CREATE (t3:Trajet {id: 'TRJ003', origine: 'E001', distance: 20, duree: 40, cout: 180})-[:PASSED_BY]->(z4)")
            conn.execute_query("MATCH (e3:Entrepôt {id: 'E003'}), (z3:Zone {id: 'Z003'}) CREATE (t4:Trajet {id: 'TRJ004', origine: 'E003', distance: 10, duree: 20, cout: 90})-[:PASSED_BY]->(z3)")

            st.success("Données de test générées avec succès. Actualisation de la page...")
            st.cache_resource.clear()
            st.experimental_rerun()

        except Exception as e:
            st.error(f"Erreur lors de la génération des données de test : {str(e)}")



# Note: La fermeture de la connexion Neo4j est gérée par st.cache_resource.
# Elle sera fermée automatiquement lorsque l'application Streamlit s'arrêtera.
