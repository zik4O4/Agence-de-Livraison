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


# Fonctions utilitaires pour les nouvelles fonctionnalités
def create_complete_order(cmd_id, client_id, date, address, products_str, livreur_id=None):
    """Crée une commande complète avec ses produits et relations"""
    try:
        # Créer la commande
        order_data = {
            "id": cmd_id,
            "nom": f"Commande {cmd_id}",
            "client_id": client_id,
            "date": date.strftime("%Y-%m-%d"),
            "statut": "Confirmée",
            "adresse_livraison": address
        }
        
        create_entity("Commande", order_data)
        
        # Traiter les produits
        total_amount = 0
        for line in products_str.strip().split('\n'):
            if ':' in line:
                prod_id, quantity = line.strip().split(':')
                quantity = int(quantity)
                
                # Créer relation CONTAINS entre commande et produit
                create_relationship("Commande", cmd_id, "Produit", prod_id, 
                                  "CONTAINS", {"quantite": quantity})
                
                # Calculer le montant (nécessiterait une requête pour le prix)
                # total_amount += get_product_price(prod_id) * quantity
        
        # Assigner livreur si spécifié
        if livreur_id:
            create_relationship("Livreur", livreur_id, "Commande", cmd_id, "ASSIGNED_TO")
        
        # Créer relation client-commande
        create_relationship("Client", client_id, "Commande", cmd_id, "ORDERED")
        
        st.success(f"Commande {cmd_id} créée avec succès!")
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la commande: {str(e)}")

def display_general_stats():
    """Affiche des statistiques générales sur les données"""
    try:
        with driver.session() as session:
            # Compter les nœuds par type
            query = """
            MATCH (n)
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
            """
            result = session.run(query)
            
            st.subheader("Statistiques des Entités")
            stats_data = []
            for record in result:
                stats_data.append({
                    "Type": record["type"],
                    "Nombre": record["count"]
                })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data))
            else:
                st.info("Aucune donnée trouvée")
                
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des statistiques: {str(e)}")


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
                    ["Client", "Produit", "Livreur", "Entrepôt", "Zone", "Commande", "Trajet"])
                
                entity_id = st.text_input("ID Unique*")
                entity_name = st.text_input("Nom*")
                
                # Champs spécifiques par type
                if entity_type == "Client":
                    client_type = st.selectbox("Type Client", ["Particulier", "Entreprise"])
                    client_address = st.text_input("Adresse")
                    client_phone = st.text_input("Téléphone")
                    
                elif entity_type == "Produit":
                    product_cat = st.text_input("Catégorie*")
                    product_price = st.number_input("Prix*", min_value=0.0)
                    product_weight = st.number_input("Poids (kg)", min_value=0.0)
                    product_stock = st.number_input("Stock", min_value=0)
                    
                elif entity_type == "Livreur":
                    livreur_phone = st.text_input("Téléphone")
                    livreur_vehicle = st.selectbox("Véhicule", ["Moto", "Voiture", "Camionnette", "Vélo"])
                    livreur_zone = st.text_input("Zone de livraison")
                    
                elif entity_type == "Entrepôt":
                    entrepot_address = st.text_input("Adresse*")
                    entrepot_capacity = st.number_input("Capacité", min_value=0)
                    
                elif entity_type == "Zone":
                    zone_city = st.text_input("Ville")
                    zone_postal = st.text_input("Code postal")
                    
                elif entity_type == "Commande":
                    st.markdown("**Informations Commande**")
                    order_client_id = st.text_input("ID Client*")
                    order_date = st.date_input("Date de commande")
                    order_status = st.selectbox("Statut", 
                        ["En attente", "Confirmée", "En préparation", "Expédiée", "Livrée", "Annulée"])
                    order_total = st.number_input("Montant total", min_value=0.0)
                    order_address = st.text_input("Adresse de livraison*")
                    
                elif entity_type == "Trajet":
                    st.markdown("**Informations Trajet**")
                    trajet_livreur_id = st.text_input("ID Livreur*")
                    trajet_date = st.date_input("Date du trajet")
                    trajet_start = st.text_input("Point de départ*")
                    trajet_end = st.text_input("Point d'arrivée*")
                    trajet_distance = st.number_input("Distance (km)", min_value=0.0)
                    trajet_duration = st.number_input("Durée estimée (minutes)", min_value=0)
                    trajet_status = st.selectbox("Statut trajet", 
                        ["Planifié", "En cours", "Terminé", "Annulé"])
                
                if st.form_submit_button(f"Créer {entity_type}"):
                    if not entity_id or not entity_name:
                        st.warning("Les champs obligatoires (*) doivent être remplis")
                    else:
                        # Construction du dictionnaire de données selon le type
                        entity_data = {"id": entity_id, "nom": entity_name}
                        
                        if entity_type == "Client":
                            entity_data.update({
                                "type": client_type,
                                "adresse": client_address,
                                "telephone": client_phone
                            })
                        elif entity_type == "Produit":
                            entity_data.update({
                                "categorie": product_cat,
                                "prix": product_price,
                                "poids": product_weight,
                                "stock": product_stock
                            })
                        elif entity_type == "Livreur":
                            entity_data.update({
                                "telephone": livreur_phone,
                                "vehicule": livreur_vehicle,
                                "zone": livreur_zone
                            })
                        elif entity_type == "Entrepôt":
                            entity_data.update({
                                "adresse": entrepot_address,
                                "capacite": entrepot_capacity
                            })
                        elif entity_type == "Zone":
                            entity_data.update({
                                "ville": zone_city,
                                "code_postal": zone_postal
                            })
                        elif entity_type == "Commande":
                            if not order_client_id or not order_address:
                                st.warning("ID Client et adresse de livraison obligatoires")
                            else:
                                entity_data.update({
                                    "client_id": order_client_id,
                                    "date": order_date.strftime("%Y-%m-%d"),
                                    "statut": order_status,
                                    "montant_total": order_total,
                                    "adresse_livraison": order_address
                                })
                        elif entity_type == "Trajet":
                            if not trajet_livreur_id or not trajet_start or not trajet_end:
                                st.warning("ID Livreur, point de départ et d'arrivée obligatoires")
                            else:
                                entity_data.update({
                                    "livreur_id": trajet_livreur_id,
                                    "date": trajet_date.strftime("%Y-%m-%d"),
                                    "point_depart": trajet_start,
                                    "point_arrivee": trajet_end,
                                    "distance": trajet_distance,
                                    "duree": trajet_duration,
                                    "statut": trajet_status
                                })
                        
                        # Créer l'entité seulement si toutes les validations sont passées
                        if entity_type not in ["Commande", "Trajet"] or \
                           (entity_type == "Commande" and order_client_id and order_address) or \
                           (entity_type == "Trajet" and trajet_livreur_id and trajet_start and trajet_end):
                            create_entity(entity_type, entity_data)

        with col2:
            with st.form("Nouvelle Relation"):
                st.subheader("Créer une Relation")
                rel_types = [
                    "LOCATED_IN", "STOCKED_IN", "DELIVERS", "ORDERED", 
                    "ASSIGNED_TO", "CONTAINS", "FOLLOWS", "SERVES"
                ]
                rel_type = st.selectbox("Type de relation*", rel_types)
                
                st.markdown("**Nœud Source**")
                from_type = st.selectbox("Type source", 
                    ["Client", "Produit", "Livreur", "Commande", "Trajet"])
                from_id = st.text_input("ID source*")
                
                st.markdown("**Nœud Cible**")
                to_type = st.selectbox("Type cible", 
                    ["Zone", "Entrepôt", "Commande", "Trajet", "Client", "Livreur"])
                to_id = st.text_input("ID cible*")
                
                # Propriétés de relation optionnelles
                st.markdown("**Propriétés de relation (optionnel)**")
                rel_props = {}
                if rel_type == "CONTAINS":
                    quantity = st.number_input("Quantité", min_value=0)
                    if quantity > 0:
                        rel_props["quantite"] = quantity
                elif rel_type == "FOLLOWS":
                    order_seq = st.number_input("Ordre de séquence", min_value=1)
                    rel_props["ordre"] = order_seq
                
                if st.form_submit_button("Établir Relation"):
                    if not from_id or not to_id:
                        st.warning("IDs source et cible obligatoires")
                    else:
                        create_relationship(from_type, from_id, to_type, to_id, rel_type, rel_props)
    
    # Section 2: Import CSV
    with st.expander("Import par Fichier CSV", expanded=True):
        st.info("Format requis: Fichier CSV avec colonnes correspondant aux propriétés des nœuds")
        
        import_type = st.selectbox("Type d'import", 
            ["Clients", "Produits", "Livreurs", "Commandes", "Trajets", "Relations"])
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        
        if uploaded_file:
            # Aperçu du fichier
            df = pd.read_csv(uploaded_file)
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            # Validation des colonnes selon le type
            required_columns = {
                "Clients": ["id", "nom"],
                "Produits": ["id", "nom", "categorie", "prix"],
                "Livreurs": ["id", "nom"],
                "Commandes": ["id", "client_id", "date", "statut", "adresse_livraison"],
                "Trajets": ["id", "livreur_id", "date", "point_depart", "point_arrivee"],
                "Relations": ["from_type", "from_id", "to_type", "to_id", "relation_type"]
            }
            
            missing_cols = set(required_columns[import_type]) - set(df.columns)
            if missing_cols:
                st.error(f"Colonnes manquantes: {missing_cols}")
            else:
                if st.button("Lancer l'Import"):
                    process_csv_import(uploaded_file, import_type)

    # Section 3: Création rapide de commandes avec produits
    with st.expander("Créer Commande Complète", expanded=False):
        st.subheader("Créer une commande avec produits")
        
        with st.form("Commande Complète"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Informations Commande**")
                cmd_id = st.text_input("ID Commande*")
                cmd_client = st.text_input("ID Client*")
                cmd_date = st.date_input("Date")
                cmd_address = st.text_input("Adresse de livraison*")
                
            with col2:
                st.markdown("**Produits**")
                products_data = st.text_area(
                    "Produits (format: id_produit:quantité, un par ligne)",
                    placeholder="PROD001:2\nPROD002:1\nPROD003:3"
                )
                
                livreur_assign = st.text_input("ID Livreur assigné (optionnel)")
            
            if st.form_submit_button("Créer Commande Complète"):
                if not cmd_id or not cmd_client or not cmd_address:
                    st.warning("Champs obligatoires manquants")
                elif not products_data:
                    st.warning("Ajoutez au moins un produit")
                else:
                    create_complete_order(cmd_id, cmd_client, cmd_date, cmd_address, 
                                        products_data, livreur_assign)

    # Section 4: Prévisualisation données existantes
    with st.expander("Vérifier les Données Existantes"):
        entity_to_check = st.selectbox("Voir tous les", 
            ["Clients", "Produits", "Livreurs", "Commandes", "Trajets", "Entrepôts", "Zones"])
        
        if st.button("Afficher"):
            display_entities(entity_to_check[:-1])  # Retire le 's' final
            
        # Statistiques rapides
        if st.button("Statistiques Générales"):
            display_general_stats()


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
                st.rerun()  # Recharge l'application pour refléter les changements
            else:
                st.error("La connexion à la base de données n'est pas établie.")
        except Exception as e:
            st.error(f"Erreur lors du nettoyage : {e}")

        

    st.markdown("---")

    st.subheader("Génération de Données de Test")
    st.info("Permet de recréer un jeu de données étendu pour les démonstrations ou le développement.")
    

    if st.button("Générer des Données de Test"):
        try:
            # Suppression des données existantes
            conn.execute_query("MATCH (n) DETACH DELETE n")

            # Création des clients (10 clients: 5 particuliers, 5 entreprises)
            conn.execute_query("CREATE (c1:Client {id: 'C001', nom: 'Martin Dupont', type: 'Particulier', email: 'martin.dupont@email.com', telephone: '0123456789'})")
            conn.execute_query("CREATE (c2:Client {id: 'C002', nom: 'TechCorp SARL', type: 'Entreprise', email: 'contact@techcorp.com', telephone: '0123456790'})")
            conn.execute_query("CREATE (c3:Client {id: 'C003', nom: 'Sophie Bernard', type: 'Particulier', email: 'sophie.bernard@email.com', telephone: '0123456791'})")
            conn.execute_query("CREATE (c4:Client {id: 'C004', nom: 'InnovatePlus SA', type: 'Entreprise', email: 'commandes@innovateplus.com', telephone: '0123456792'})")
            conn.execute_query("CREATE (c5:Client {id: 'C005', nom: 'Jean Moreau', type: 'Particulier', email: 'jean.moreau@email.com', telephone: '0123456793'})")
            conn.execute_query("CREATE (c6:Client {id: 'C006', nom: 'LogiServices Ltd', type: 'Entreprise', email: 'orders@logiservices.com', telephone: '0123456794'})")
            conn.execute_query("CREATE (c7:Client {id: 'C007', nom: 'Marie Dubois', type: 'Particulier', email: 'marie.dubois@email.com', telephone: '0123456795'})")
            conn.execute_query("CREATE (c8:Client {id: 'C008', nom: 'MegaStore Inc', type: 'Entreprise', email: 'procurement@megastore.com', telephone: '0123456796'})")
            conn.execute_query("CREATE (c9:Client {id: 'C009', nom: 'Pierre Leroy', type: 'Particulier', email: 'pierre.leroy@email.com', telephone: '0123456797'})")
            conn.execute_query("CREATE (c10:Client {id: 'C010', nom: 'DigitalFlow SAS', type: 'Entreprise', email: 'achats@digitalflow.com', telephone: '0123456798'})")

            # Création des livreurs (6 livreurs avec différents véhicules)
            conn.execute_query("CREATE (l1:Livreur {id: 'L001', nom: 'Dupont', prenom: 'Paul', vehicule: 'Moto', experience: 3, zone_preferee: 'Centre'})")
            conn.execute_query("CREATE (l2:Livreur {id: 'L002', nom: 'Martin', prenom: 'Luc', vehicule: 'Camionnette', experience: 5, zone_preferee: 'Ouest'})")
            conn.execute_query("CREATE (l3:Livreur {id: 'L003', nom: 'Bernard', prenom: 'Alex', vehicule: 'Vélo', experience: 1, zone_preferee: 'Centre'})")
            conn.execute_query("CREATE (l4:Livreur {id: 'L004', nom: 'Dubois', prenom: 'Marc', vehicule: 'Camionnette', experience: 7, zone_preferee: 'Industrielle'})")
            conn.execute_query("CREATE (l5:Livreur {id: 'L005', nom: 'Leroy', prenom: 'Tom', vehicule: 'Camion', experience: 10, zone_preferee: 'Périphérie'})")
            conn.execute_query("CREATE (l6:Livreur {id: 'L006', nom: 'Rousseau', prenom: 'Julie', vehicule: 'Moto', experience: 2, zone_preferee: 'Résidentielle'})")

            # Création des entrepôts (5 entrepôts avec différentes capacités)
            conn.execute_query("CREATE (e1:Entrepôt {id: 'E001', nom: 'Entrepôt Nord', capacite: 1000, adresse: '15 Rue du Commerce Nord', specialite: 'Électronique'})")
            conn.execute_query("CREATE (e2:Entrepôt {id: 'E002', nom: 'Entrepôt Sud', capacite: 1500, adresse: '25 Avenue du Sud', specialite: 'Général'})")
            conn.execute_query("CREATE (e3:Entrepôt {id: 'E003', nom: 'Entrepôt Ouest', capacite: 800, adresse: '10 Boulevard Ouest', specialite: 'Électroménager'})")
            conn.execute_query("CREATE (e4:Entrepôt {id: 'E004', nom: 'Entrepôt Central', capacite: 2000, adresse: '5 Place Centrale', specialite: 'Multimédia'})")
            conn.execute_query("CREATE (e5:Entrepôt {id: 'E005', nom: 'Entrepôt Est', capacite: 1200, adresse: '30 Rue de l\\'Est', specialite: 'Livres'})")

            # Création des produits (15 produits dans 5 catégories)
            # Catégorie Électronique
            conn.execute_query("CREATE (p1:Produit {id: 'P001', nom: 'Ordinateur Portable', categorie: 'Électronique', prix: 1200, poids: 2.5, stock_minimum: 10})")
            conn.execute_query("CREATE (p2:Produit {id: 'P002', nom: 'Smartphone', categorie: 'Électronique', prix: 800, poids: 0.3, stock_minimum: 20})")
            conn.execute_query("CREATE (p3:Produit {id: 'P003', nom: 'Tablette', categorie: 'Électronique', prix: 450, poids: 0.7, stock_minimum: 15})")
            
            # Catégorie Électroménager
            conn.execute_query("CREATE (p4:Produit {id: 'P004', nom: 'Cafetière', categorie: 'Électroménager', prix: 80, poids: 1.2, stock_minimum: 25})")
            conn.execute_query("CREATE (p5:Produit {id: 'P005', nom: 'Grille-pain', categorie: 'Électroménager', prix: 45, poids: 1.8, stock_minimum: 30})")
            conn.execute_query("CREATE (p6:Produit {id: 'P006', nom: 'Aspirateur', categorie: 'Électroménager', prix: 220, poids: 4.5, stock_minimum: 12})")
            
            # Catégorie Multimédia
            conn.execute_query("CREATE (p7:Produit {id: 'P007', nom: 'Casque Audio', categorie: 'Multimédia', prix: 150, poids: 0.2, stock_minimum: 40})")
            conn.execute_query("CREATE (p8:Produit {id: 'P008', nom: 'Enceinte Bluetooth', categorie: 'Multimédia', prix: 90, poids: 1.0, stock_minimum: 35})")
            conn.execute_query("CREATE (p9:Produit {id: 'P009', nom: 'Webcam HD', categorie: 'Multimédia', prix: 75, poids: 0.4, stock_minimum: 50})")
            
            # Catégorie Livre
            conn.execute_query("CREATE (p10:Produit {id: 'P010', nom: 'Livre \"Neo4j Guide\"', categorie: 'Livre', prix: 30, poids: 0.5, stock_minimum: 100})")
            conn.execute_query("CREATE (p11:Produit {id: 'P011', nom: 'Manuel Python', categorie: 'Livre', prix: 45, poids: 0.8, stock_minimum: 80})")
            conn.execute_query("CREATE (p12:Produit {id: 'P012', nom: 'Guide JavaScript', categorie: 'Livre', prix: 35, poids: 0.6, stock_minimum: 90})")
            
            # Catégorie Accessoires
            conn.execute_query("CREATE (p13:Produit {id: 'P013', nom: 'Souris sans fil', categorie: 'Accessoires', prix: 25, poids: 0.1, stock_minimum: 60})")
            conn.execute_query("CREATE (p14:Produit {id: 'P014', nom: 'Clavier mécanique', categorie: 'Accessoires', prix: 120, poids: 1.1, stock_minimum: 25})")
            conn.execute_query("CREATE (p15:Produit {id: 'P015', nom: 'Support ordinateur', categorie: 'Accessoires', prix: 60, poids: 2.0, stock_minimum: 20})")

            # Création des zones (8 zones géographiques)
            conn.execute_query("CREATE (z1:Zone {id: 'Z001', nom: 'Centre Ville', densite_population: 'Élevée', code_postal: '75001'})")
            conn.execute_query("CREATE (z2:Zone {id: 'Z002', nom: 'Banlieue Ouest', densite_population: 'Moyenne', code_postal: '92000'})")
            conn.execute_query("CREATE (z3:Zone {id: 'Z003', nom: 'Zone Industrielle', densite_population: 'Faible', code_postal: '93000'})")
            conn.execute_query("CREATE (z4:Zone {id: 'Z004', nom: 'Quartier Résidentiel', densite_population: 'Moyenne', code_postal: '94000'})")
            conn.execute_query("CREATE (z5:Zone {id: 'Z005', nom: 'Banlieue Est', densite_population: 'Élevée', code_postal: '77000'})")
            conn.execute_query("CREATE (z6:Zone {id: 'Z006', nom: 'Quartier Nord', densite_population: 'Moyenne', code_postal: '95000'})")
            conn.execute_query("CREATE (z7:Zone {id: 'Z007', nom: 'Périphérie Sud', densite_population: 'Faible', code_postal: '91000'})")
            conn.execute_query("CREATE (z8:Zone {id: 'Z008', nom: 'Zone Commerciale', densite_population: 'Élevée', code_postal: '78000'})")

            # Création des relations LOCATED_IN (clients dans les zones)
            conn.execute_query("MATCH (c1:Client {id: 'C001'}), (z1:Zone {id: 'Z001'}) CREATE (c1)-[:LOCATED_IN]->(z1)")
            conn.execute_query("MATCH (c2:Client {id: 'C002'}), (z2:Zone {id: 'Z002'}) CREATE (c2)-[:LOCATED_IN]->(z2)")
            conn.execute_query("MATCH (c3:Client {id: 'C003'}), (z3:Zone {id: 'Z003'}) CREATE (c3)-[:LOCATED_IN]->(z3)")
            conn.execute_query("MATCH (c4:Client {id: 'C004'}), (z4:Zone {id: 'Z004'}) CREATE (c4)-[:LOCATED_IN]->(z4)")
            conn.execute_query("MATCH (c5:Client {id: 'C005'}), (z5:Zone {id: 'Z005'}) CREATE (c5)-[:LOCATED_IN]->(z5)")
            conn.execute_query("MATCH (c6:Client {id: 'C006'}), (z6:Zone {id: 'Z006'}) CREATE (c6)-[:LOCATED_IN]->(z6)")
            conn.execute_query("MATCH (c7:Client {id: 'C007'}), (z7:Zone {id: 'Z007'}) CREATE (c7)-[:LOCATED_IN]->(z7)")
            conn.execute_query("MATCH (c8:Client {id: 'C008'}), (z8:Zone {id: 'Z008'}) CREATE (c8)-[:LOCATED_IN]->(z8)")
            conn.execute_query("MATCH (c9:Client {id: 'C009'}), (z1:Zone {id: 'Z001'}) CREATE (c9)-[:LOCATED_IN]->(z1)")
            conn.execute_query("MATCH (c10:Client {id: 'C010'}), (z2:Zone {id: 'Z002'}) CREATE (c10)-[:LOCATED_IN]->(z2)")

            # Création des relations ASSIGNED_TO (livreurs assignés aux zones)
            conn.execute_query("MATCH (l1:Livreur {id: 'L001'}), (z1:Zone {id: 'Z001'}) CREATE (l1)-[:ASSIGNED_TO]->(z1)")
            conn.execute_query("MATCH (l2:Livreur {id: 'L002'}), (z2:Zone {id: 'Z002'}) CREATE (l2)-[:ASSIGNED_TO]->(z2)")
            conn.execute_query("MATCH (l3:Livreur {id: 'L003'}), (z1:Zone {id: 'Z001'}) CREATE (l3)-[:ASSIGNED_TO]->(z1)")
            conn.execute_query("MATCH (l4:Livreur {id: 'L004'}), (z3:Zone {id: 'Z003'}) CREATE (l4)-[:ASSIGNED_TO]->(z3)")
            conn.execute_query("MATCH (l5:Livreur {id: 'L005'}), (z7:Zone {id: 'Z007'}) CREATE (l5)-[:ASSIGNED_TO]->(z7)")
            conn.execute_query("MATCH (l6:Livreur {id: 'L006'}), (z4:Zone {id: 'Z004'}) CREATE (l6)-[:ASSIGNED_TO]->(z4)")

            # Création des relations STOCKED_IN (produits dans les entrepôts)
            stock_relations = [
                ('P001', 'E001', 50), ('P002', 'E001', 75), ('P003', 'E001', 60),
                ('P004', 'E003', 100), ('P005', 'E003', 85), ('P006', 'E003', 40),
                ('P007', 'E004', 120), ('P008', 'E004', 95), ('P009', 'E004', 80),
                ('P010', 'E005', 200), ('P011', 'E005', 150), ('P012', 'E005', 180),
                ('P013', 'E002', 200), ('P014', 'E002', 60), ('P015', 'E002', 45)
            ]
            
            for produit_id, entrepot_id, quantite in stock_relations:
                conn.execute_query(f"MATCH (p:Produit {{id: '{produit_id}'}}), (e:Entrepôt {{id: '{entrepot_id}'}}) CREATE (p)-[:STOCKED_IN {{quantite: {quantite}}}]->(e)")

            # Fonction pour créer des commandes
            def create_commande(cmd_id, client_id, livreur_id, produits, date_cmd, statut, prix_total, poids_total):
                conn.execute_query(f"""
                MATCH (c:Client {{id: '{client_id}'}}), (l:Livreur {{id: '{livreur_id}'}})
                CREATE (cmd:Commande {{
                    id: '{cmd_id}', 
                    date_commande: date('{date_cmd}'), 
                    prix_total: {prix_total}, 
                    poids_total: {poids_total}, 
                    statut: '{statut}'
                }})
                CREATE (c)-[:ORDERED]->(cmd)
                CREATE (l)-[:DELIVERS]->(cmd)
                """)
                
                for produit_id, quantite in produits:
                    conn.execute_query(f"""
                    MATCH (cmd:Commande {{id: '{cmd_id}'}}), (p:Produit {{id: '{produit_id}'}})
                    CREATE (cmd)-[:CONTAINS {{quantite: {quantite}}}]->(p)
                    """)

            # Création des 32 commandes avec différents statuts
            commandes = [
                ('CMD001', 'C001', 'L001', [('P001', 1), ('P010', 1)], '2023-01-15', 'Livré', 1230, 3.0),
                ('CMD002', 'C002', 'L002', [('P004', 2)], '2023-01-16', 'En cours', 160, 2.4),
                ('CMD003', 'C003', 'L004', [('P002', 1), ('P007', 1)], '2023-01-17', 'Livré', 950, 0.5),
                ('CMD004', 'C004', 'L001', [('P003', 3)], '2023-01-18', 'En attente', 1350, 2.1),
                ('CMD005', 'C005', 'L003', [('P013', 2), ('P010', 1)], '2023-01-19', 'Livré', 80, 0.7),
                ('CMD006', 'C006', 'L005', [('P006', 1), ('P005', 1)], '2023-01-20', 'Expédié', 265, 6.3),
                ('CMD007', 'C007', 'L006', [('P008', 1)], '2023-01-21', 'Livré', 90, 1.0),
                ('CMD008', 'C008', 'L002', [('P001', 2), ('P014', 1)], '2023-01-22', 'En cours', 2520, 6.1),
                ('CMD009', 'C009', 'L001', [('P009', 1), ('P011', 1)], '2023-01-23', 'Livré', 120, 1.2),
                ('CMD010', 'C010', 'L004', [('P012', 2)], '2023-01-24', 'En attente', 70, 1.2),
                ('CMD011', 'C001', 'L003', [('P015', 1)], '2023-01-25', 'Expédié', 60, 2.0),
                ('CMD012', 'C002', 'L005', [('P004', 1), ('P005', 1)], '2023-01-26', 'Livré', 125, 3.0),
                ('CMD013', 'C003', 'L006', [('P002', 1)], '2023-01-27', 'En cours', 800, 0.3),
                ('CMD014', 'C004', 'L001', [('P007', 2), ('P013', 1)], '2023-01-28', 'Livré', 325, 0.5),
                ('CMD015', 'C005', 'L002', [('P001', 1)], '2023-01-29', 'En attente', 1200, 2.5),
                ('CMD016', 'C006', 'L004', [('P003', 1), ('P008', 1)], '2023-01-30', 'Expédié', 540, 1.7),
                ('CMD017', 'C007', 'L003', [('P010', 3)], '2023-02-01', 'Livré', 90, 1.5),
                ('CMD018', 'C008', 'L005', [('P006', 1)], '2023-02-02', 'En cours', 220, 4.5),
                ('CMD019', 'C009', 'L006', [('P009', 2), ('P011', 1)], '2023-02-03', 'Livré', 195, 1.6),
                ('CMD020', 'C010', 'L001', [('P012', 1), ('P014', 1)], '2023-02-04', 'En attente', 155, 1.7),
                ('CMD021', 'C001', 'L002', [('P015', 1), ('P013', 2)], '2023-02-05', 'Expédié', 110, 2.2),
                ('CMD022', 'C002', 'L004', [('P004', 3)], '2023-02-06', 'Livré', 240, 3.6),
                ('CMD023', 'C003', 'L003', [('P002', 1), ('P007', 1)], '2023-02-07', 'En cours', 950, 0.5),
                ('CMD024', 'C004', 'L005', [('P001', 1), ('P008', 1)], '2023-02-08', 'Livré', 1290, 3.5),
                ('CMD025', 'C005', 'L006', [('P005', 2)], '2023-02-09', 'En attente', 90, 3.6),
                ('CMD026', 'C006', 'L001', [('P006', 1), ('P009', 1)], '2023-02-10', 'Expédié', 295, 4.9),
                ('CMD027', 'C007', 'L002', [('P010', 1), ('P011', 1)], '2023-02-11', 'Livré', 75, 1.3),
                ('CMD028', 'C008', 'L004', [('P003', 2)], '2023-02-12', 'En cours', 900, 1.4),
                ('CMD029', 'C009', 'L003', [('P012', 1), ('P013', 1)], '2023-02-13', 'Livré', 60, 0.7),
                ('CMD030', 'C010', 'L005', [('P014', 1)], '2023-02-14', 'En attente', 120, 1.1),
                ('CMD031', 'C001', 'L006', [('P015', 1), ('P007', 1)], '2023-02-15', 'Expédié', 210, 2.2),
                ('CMD032', 'C002', 'L001', [('P001', 1), ('P004', 1)], '2023-02-16', 'Livré', 1280, 3.7)
            ]
            
            for cmd_data in commandes:
                create_commande(*cmd_data)

            # Création des trajets
            trajets = [
                ('TRJ001', 'E001', 'Z001', 15, 30, 150),
                ('TRJ002', 'E002', 'Z002', 25, 45, 200),
                ('TRJ003', 'E001', 'Z004', 20, 40, 180),
                ('TRJ004', 'E003', 'Z003', 10, 20, 90),
                ('TRJ005', 'E004', 'Z001', 18, 35, 160),
                ('TRJ006', 'E005', 'Z005', 22, 42, 190),
                ('TRJ007', 'E002', 'Z006', 28, 50, 220),
                ('TRJ008', 'E003', 'Z007', 35, 60, 280),
                ('TRJ009', 'E004', 'Z008', 12, 25, 120),
                ('TRJ010', 'E001', 'Z005', 30, 55, 250)
            ]
            
            for trajet_id, entrepot_id, zone_id, distance, duree, cout in trajets:
                conn.execute_query(f"MATCH (e:Entrepôt {{id: '{entrepot_id}'}}), (z:Zone {{id: '{zone_id}'}}) CREATE (t:Trajet {{id: '{trajet_id}', origine: '{entrepot_id}', distance: {distance}, duree: {duree}, cout: {cout}}})-[:PASSED_BY]->(z)")

            st.success("Données de test étendues générées avec succès. Actualisation de la page...")
            st.cache_resource.clear()
            st.rerun()

        except Exception as e:
            st.error(f"Erreur lors de la génération des données de test : {str(e)}")
# Note: La fermeture de la connexion Neo4j est gérée par st.cache_resource.
# Elle sera fermée automatiquement lorsque l'application Streamlit s'arrêtera.

    if st.button("Générer des Données de Test 2"):
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
            st.rerun()

        except Exception as e:
            st.error(f"Erreur lors de la génération des données de test : {str(e)}")



