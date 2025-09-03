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
from azure.cosmos import CosmosClient, PartitionKey
import math
import sys
import os
import time
from typing import Iterable, Tuple, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Base endpoint from Cosmos DB portal
SQL_ENDPOINT = 'https://nwang.documents.azure.com'
# Gremlin endpoint uses WebSocket protocol with .gremlin.cosmos.azure.com suffix
GREMLIN_ENDPOINT = 'wss://nwang.gremlin.cosmos.azure.com:443'
DATABASE = 'ICIO'

# Setup helpers
def check_graph(
    graph: str,
    partition_key_field: str,
    throughput: int,
):
    """
    Ensures the Cosmos DB database and Gremlin graph (container) exist.
    Uses azure-cosmos SDK (SQL API) since Gremlin is provisioned on same account.
    """
    key = os.getenv("PRIMARY_KEY")
    cosmos = CosmosClient(SQL_ENDPOINT, credential=key)

    # Create database if not exists
    db = cosmos.get_database_client(DATABASE)

    # Create container (graph) if not exists
    db.create_container_if_not_exists(
        id=graph,
        partition_key=PartitionKey(path=f"/{partition_key_field}"),
        offer_throughput=throughput,
    )
    print(f"Graph '{graph}' is ready in database '{DATABASE}'.")

# Gremlin helpers
def build_client(graph: str) -> Client:
    primary_key = os.getenv("PRIMARY_KEY")
    return Client(
        GREMLIN_ENDPOINT,
        "g",
        username=f"/dbs/{DATABASE}/colls/{graph}",
        password=primary_key,
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
def read_wide_icio(csv_path: str, index_col: str = "V1", out_col: str = "OUT"):
    df = pd.read_csv(csv_path, index_col=index_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    all_cols = list(df.columns)
    sector_cols = [c for c in all_cols if str(c) != out_col]
    return df, sector_cols


# Main ETL logic
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--index-col", default="V1")
    ap.add_argument("--out-col", default="OUT")

    ap.add_argument("--graph", required=True)

    ap.add_argument("--partition-key-field", default="pk")
    ap.add_argument("--throughput", type=int, default=400)
    ap.add_argument(
        "--min-value", type=float, default=1e-9, help="Drop flows smaller than this"
    )
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--skip-vertices", action="store_true", help="Skip upserting vertices")
    args = ap.parse_args()

    print("Reading CSV...")
    df, sector_cols = read_wide_icio(
        args.csv, index_col=args.index_col, out_col=args.out_col
    )
    print(f"Found {len(df)} supplier rows, {len(sector_cols)} target columns")

    suppliers = list(df.index.astype(str))
    targets = [str(c) for c in sector_cols]
    all_nodes = sorted(set(suppliers) | set(targets))

    def node_to_country(node_id: str):
        if "_" in node_id:
            return node_id.split("_", 1)[0]
        return node_id

    def node_label(node_id: str):
        # classify node
        if "_" not in node_id:
            return "sector"
        _, suffix = node_id.split("_", 1)
        # OECD FD categories: HFCE, NPISH, GGFC, GFCF, INVNT, DPABR
        if suffix in {"HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"}:
            return "final_demand"
        return "sector"

    # Prepare vertex records
    v_records = {}
    for n in all_nodes:
        country = node_to_country(n)
        v_records[n] = {
            "label": node_label(n),
            args.partition_key_field: country,
            "country": country,
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
        f"Prepared {len(v_records)} vertices and {len(edges)} edges (including final demand)."
    )

    # Connect to Cosmos
    print("Validating graph status for update...")
    check_graph(args.graph, args.partition_key_field, args.throughput)

    print("Connecting to Cosmos Gremlin endpoint...")
    client = build_client(args.graph)

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

    print("Done.")

if __name__ == "__main__":
    main()
