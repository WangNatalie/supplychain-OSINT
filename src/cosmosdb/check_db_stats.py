#!/usr/bin/env python3
"""
check_db_stats.py

Check database statistics for Cosmos DB Gremlin graphs.
Can be called independently or imported by other scripts.

Usage:
python check_db_stats.py --graph "<graph_name>"
"""

import argparse
import os
from gremlin_python.driver.client import Client, serializer
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

load_dotenv()

# Cosmos DB endpoint
SQL_ENDPOINT = 'https://graph-db-osint.documents.azure.com:443/'
# Gremlin endpoint
GREMLIN_ENDPOINT = 'wss://graph-db-osint.gremlin.cosmos.azure.com:443'
DATABASE = 'osint-db'

def check_graph_exists(graph: str, partition_key_field: str = "pk", throughput: int = 400):
    """
    Ensures the Cosmos DB database and Gremlin graph (container) exist.
    Uses azure-cosmos SDK (SQL API) since Gremlin is provisioned on same account.
    """
    key = os.getenv("ACCOUNT_KEY")
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

def build_client(graph: str) -> Client:
    """Build Gremlin client for the specified graph."""
    ACCOUNT_KEY = os.getenv("ACCOUNT_KEY")
    return Client(
        GREMLIN_ENDPOINT,
        "g",
        username=f"/dbs/{DATABASE}/colls/{graph}",
        password=ACCOUNT_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0()
    )

def get_graph_stats(client: Client):
    """
    Get current node and edge counts from the graph.
    Returns (vertex_count, edge_count) or (None, None) if error.
    """
    try:
        # Count vertices
        vertex_count = client.submit("g.V().count()").all().result()[0]
        
        # Count edges
        edge_count = client.submit("g.E().count()").all().result()[0]
        
        return vertex_count, edge_count
    except Exception as e:
        print(f"Warning: Could not retrieve graph statistics: {e}")
        return None, None

def get_existing_vertex_ids(client: Client, vertex_ids: list) -> set:
    """Get set of vertex IDs that already exist in the graph"""
    try:
        # Query all existing vertex IDs
        result = client.submit("g.V().id()").all().result()
        return set(result)
    except Exception as e:
        print(f"Warning: Could not check existing vertices: {e}")
        return set()

def check_database_stats(graph: str, partition_key_field: str = "pk", throughput: int = 1000):
    """
    Check and display database statistics for the specified graph.
    
    Args:
        graph: Name of the graph/container
        partition_key_field: Partition key field name
        throughput: Database throughput
        verbose: Whether to print detailed output
    
    Returns:
        tuple: (vertex_count, edge_count) or (None, None) if error
    """

    print("Validating graph status...")
    
    check_graph_exists(graph, partition_key_field, throughput)

    print("Connecting to Cosmos Gremlin endpoint...")
    
    client = build_client(graph)
    
    try:
        print("Checking current graph statistics...")
        
        vertex_count, edge_count = get_graph_stats(client)
        
        if vertex_count is not None and edge_count is not None:
            print(f"Current graph contains {vertex_count} nodes and {edge_count} relationships")
            return vertex_count, edge_count
        else:
            print("Graph appears to be empty or statistics unavailable")
            return None, None
    finally:
        # Always close the client connection
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Error closing client connection: {e}")

def main():
    """Command line interface for checking database statistics."""
    ap = argparse.ArgumentParser(description="Check database statistics for Cosmos DB Gremlin graphs")
    ap.add_argument("--graph", required=True, help="Name of the graph/container")
    ap.add_argument("--partition-key-field", default="pk", help="Partition key field name")
    ap.add_argument("--throughput", type=int, default=1000, help="Database throughput")
    
    args = ap.parse_args()
    
    vertex_count, edge_count = check_database_stats(
        args.graph, 
        args.partition_key_field, 
        args.throughput
    )               
    
    return vertex_count, edge_count

if __name__ == "__main__":
    main()
