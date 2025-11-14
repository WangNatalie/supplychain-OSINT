#!/usr/bin/env python3
"""
icio_full_to_cosmos.py

Load OECD-style wide ICIO Z-matrix CSV into Azure Cosmos DB (Gremlin API)
as a directed graph, including both intermediate industries and final demand nodes.

Usage:
python load_icio.py \
  --csv /path/your.csv \
  --graph "<graph>" \
  --min-value 1e-6 \
  --batch-size 200
"""

import argparse
import pandas as pd
from gremlin_python.driver.client import Client, serializer
from gremlin_python.driver.protocol import GremlinServerError
import math
import sys
import os
import time
from typing import Iterable, Tuple, Dict, Any

from dotenv import load_dotenv
from check_db_stats import check_database_stats, build_client, get_existing_vertex_ids, get_graph_stats
load_dotenv()

# Base endpoint from Cosmos DB portal
SQL_ENDPOINT = 'https://graph-db-osint.documents.azure.com:443/'
# Gremlin endpoint uses WebSocket protocol with .gremlin.cosmos.azure.com suffix
GREMLIN_ENDPOINT = 'wss://graph-db-osint.gremlin.cosmos.azure.com:443'
DATABASE = 'osint-db'



# Gremlin helpers
def build_client(graph: str) -> Client:
    ACCOUNT_KEY = os.getenv("ACCOUNT_KEY")
    client = Client(
        GREMLIN_ENDPOINT,
        "g",
        username=f"/dbs/{DATABASE}/colls/{graph}",
        password=ACCOUNT_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0()
    )


def upsert_vertices_gremlin_cmd(vertex_props: Iterable[Tuple[str, Dict[str, Any]]]):
    """Generate individual Gremlin commands for vertex upserts"""
    for vid, props in vertex_props:
        label = props.get("label", "node")
        props_cmd = "".join([f".property('{k}', {repr(v)})" for k, v in props.items()])
        cmd = f"g.V('{vid}').fold().coalesce(unfold(), addV('{label}').property('id','{vid}'){props_cmd})"
        yield cmd

def upsert_edges_gremlin_cmd(edge_tuples: Iterable[Tuple[str, str, Dict[str, Any]]]):
    """Generate individual Gremlin commands for edge upserts"""
    for out_id, in_id, props in edge_tuples:
        label = props.get("label", "supplies")
        prop_str = "".join(
            [
                f".property('{k}', {repr(v)})"
                for k, v in props.items()
                if k not in {"label"} and v is not None
            ]
        )
        # Simplified query that works with Azure Cosmos DB
        cmd = (
            f"g.V('{out_id}').outE('{label}').where(inV().hasId('{in_id}')).fold()."
            f"coalesce(unfold(), g.V('{out_id}').addE('{label}').to(g.V('{in_id}')){prop_str})"
        )
        yield cmd

def chunked(iterable: Iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def retry_with_backoff(func, max_retries=5, base_delay=0.1):
    """Retry function with exponential backoff for rate limiting"""
    for attempt in range(max_retries):
        try:
            return func()
        except GremlinServerError as e:
            if e.status_code == 429 or (hasattr(e, 'status_attributes') and 
                                       e.status_attributes.get('x-ms-status-code') == 429):
                if attempt < max_retries - 1:
                    # Get retry delay from response or use exponential backoff
                    retry_after = e.status_attributes.get('x-ms-retry-after-ms', None)
                    if retry_after:
                        # Parse the time format "00:00:00.0760000" correctly
                        if retry_after.startswith('00:00:00.'):
                            # Extract milliseconds: "076" from "00:00:00.0760000"
                            ms_str = retry_after.replace('00:00:00.', '').rstrip('0')
                            delay = float(ms_str) / 1000
                        else:
                            delay = float(retry_after) / 1000
                    else:
                        delay = base_delay * (1.5 ** attempt)
                    
                    print(f"Rate limited, retrying in {delay:.3f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            raise
    return func()  # Last attempt


# CSV parsing
def read_icio(csv_path: str, index_col: str = "V1"):
    try:
        df = pd.read_csv(csv_path, index_col=index_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        all_cols = list(df.columns)
        
        # Filter out final demand categories
        final_demand_codes = ['HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR', 'OUT']
        sector_cols = []
        for c in all_cols:
            col_str = str(c)
            # Skip if it's a final demand category
            if any(fd_code in col_str for fd_code in final_demand_codes):
                continue
            # Skip if it doesn't follow the COUNTRY_INDUSTRY pattern
            if '_' not in col_str:
                continue
            sector_cols.append(c)
        
        return df, sector_cols
        
    except FileNotFoundError:
       print(f"Error: File {csv_path} not found")
       


# Main ETL logic
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--index-col", default="V1")

    ap.add_argument("--graph", required=True)

    ap.add_argument("--throughput", type=int, default=1000)
    ap.add_argument(
        "--min-value", type=float, default=1e-9, help="Drop flows smaller than this"
    )
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--skip-vertices", action="store_true", help="Skip upserting vertices")
    ap.add_argument("--dry-run", action="store_true", help="Stop after preparing vertices and edges, don't perform any upserts")
    args = ap.parse_args()

    # Connect to Cosmos for stats check
    vertex_count, edge_count = check_database_stats(args.graph, "pk", args.throughput)
    
    # Build client for data operations
    client = build_client(args.graph)

    print("Reading CSV...")
    df, sector_cols = read_icio(
        args.csv, index_col=args.index_col
    )
    print(f"Found {len(df)} supplier rows, {len(sector_cols)} target columns")

    suppliers = list(df.index.astype(str))
    targets = [str(c) for c in sector_cols]
    all_nodes = sorted(set(suppliers) | set(targets))

    def node_to_country(node_id: str):
        if "_" in node_id:
            return node_id.split("_", 1)[0]
        return node_id

    def node_to_sector(node_id: str):
        if "_" in node_id:
            return node_id.split("_", 1)[1]
        return node_id

    # Prepare vertex records
    v_records = {}
    for n in all_nodes:
        country = node_to_country(n)
        sector = node_to_sector(n)
        v_records[n] = {
            "label": "Country_sector",
            "pk": n,  # Country_sector as partition key
            "country": country,
            "sector": sector,
            "id_str": n,
        }

    # Build edges
    edges = []
    for supplier in suppliers:
        row = df.loc[supplier, sector_cols]
        row = row.dropna()
        for target, val in row.items():
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            if abs(float(val)) < args.min_value:
                continue
            edges.append((supplier, str(target), {"label": "supplies", "value": float(val)}))

    print(
        f"Prepared {len(v_records)} vertices and {len(edges)} edges (excluding final demand)."
    )

    # Exit early if dry-run mode
    if args.dry_run:
        print("Dry run mode: Stopping after preparation. No data will be upserted.")
        return

    # Upsert vertices
    if not args.skip_vertices:
        print("Upserting vertices...")
        vertex_count = 0
        for chunk in chunked(v_records.items(), args.batch_size):
            for cmd in upsert_vertices_gremlin_cmd(chunk):
                try:
                    retry_with_backoff(lambda: client.submit(cmd).all().result())
                    vertex_count += 1
                    if vertex_count % 100 == 0:
                        print(f"Inserted {vertex_count} vertices...")
                    time.sleep(0.01)
                except GremlinServerError as e:
                    print("Vertex upsert failed:", e.status_code, e.status_attributes, file=sys.stderr)
                    raise
        print(f"Completed vertex insertion: {vertex_count} vertices")
    else:
        print("Skipping vertex upserts as requested.")

    # Upsert edges
    print("Upserting edges...")
    edge_count = 0
    for chunk in chunked(edges, args.batch_size):
        for cmd in upsert_edges_gremlin_cmd(chunk):
            try:
                retry_with_backoff(lambda: client.submit(cmd).all().result())
                edge_count += 1
                if edge_count % 1000 == 0:
                    print(f"Inserted {edge_count} edges...")
                time.sleep(0.1)  # 100ms delay between requests
            except GremlinServerError as e:
                print("Edge upsert failed:", e.status_code, e.status_attributes, file=sys.stderr)
                raise

    print(f"Completed edge insertion: {edge_count} edges")

    # Show final graph statistics
    print("Checking final graph statistics...")
    final_vertex_count, final_edge_count = get_graph_stats(client)
    if final_vertex_count is not None and final_edge_count is not None:
        print(f"Final graph contains {final_vertex_count} nodes and {final_edge_count} relationships")
    else:
        print("Could not retrieve final graph statistics")

    print("Done.")

if __name__ == "__main__":
    main()
