# Delivery Logistics Graph Application

A Streamlit and Neo4j application for modeling delivery operations, managing graph entities and relationships, importing CSV data and exploring operational statistics.

## Problem and solution

Delivery data connect customers, couriers, orders, products, warehouses, geographic zones and routes. The app uses Neo4j nodes and relationships to represent these connections and exposes management/query interfaces with interactive visualizations.

**Stack:** Python, Neo4j/Cypher, Streamlit, streamlit-option-menu, pandas, Plotly, NetworkX, Matplotlib and NumPy.

## Architecture

`app.py` contains the database adapter, entity and relationship operations, import handlers, Cypher queries and UI. Neo4j provides persistent graph storage; Streamlit renders forms and dashboards. This is a development prototype with a single-file architecture.

## Installation

```bash
git clone https://github.com/zik4O4/Agence-de-Livraison.git
cd Agence-de-Livraison
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start a local Neo4j database with Bolt enabled, or use an accessible Neo4j instance. Use a dedicated development database and account. On Windows activate `.venv\Scripts\Activate.ps1`.

Create `.streamlit/secrets.toml` locally:

```toml
[neo4j]
uri = "bolt://localhost:7687"
user = "neo4j"
password = "REPLACE_WITH_YOUR_LOCAL_PASSWORD"
```

Replace the placeholder and do not commit the file. The app checks connectivity with `RETURN 1`. Providing this file also avoids relying on the original development password fallback. The current requirements are not a tested lockfile.

## Usage

```bash
streamlit run app.py
```

Begin with entity creation using the labels exposed by the app (`Client`, `Livreur`, `Commande`, `Produit`, `Entrepôt`, `Zone` and `Trajet`). Create endpoint nodes before adding relationships. An empty graph is sufficient to begin manual entity creation; no supplied seed dataset is required. Query/dashboard views become useful after data are added.

Entity CSV import requires `type_entite`, `id` and `nom` columns. Example synthetic input:

```csv
type_entite,id,nom
Client,demo-client-1,Demo Customer
```

Use the app's import guidance for extra properties and relationship formats. The source remains the authority for supported fields; validate imports against a disposable database.

## Limitations

- Some expanded workflow helpers are incomplete. For example, `create_complete_order` calls `create_relationship` with a properties argument that its current signature does not accept. This setup/documentation change does not redesign those flows.
- Database access, domain operations and presentation are mixed in one large source file.
- No production authentication, authorization, deployment benchmark or automated end-to-end verification is claimed.
- No demonstration screenshot is included; `icon.png` is an application icon.

## License

No repository-level license has been selected. Third-party library terms apply separately.
