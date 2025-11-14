#!/usr/bin/env python3
"""
cleanup_db.py

Clean up Cosmos DB Gremlin graph data.
Can clear all data or specific partitions.

Usage:
python cleanup_db.py --graph "<graph_name>"
python cleanup_db.py --graph "<graph_name>" --partition-key "country" --partition-value "USA"
"""

import argparse
import os
import time
from gremlin_python.driver.client import Client, serializer
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from check_db_stats import build_client, get_graph_stats

load_dotenv()

# Cosmos DB endpoint
SQL_ENDPOINT = 'https://graph-db-osint.documents.azure.com:443/'
# Gremlin endpoint
GREMLIN_ENDPOINT = 'wss://graph-db-osint.gremlin.cosmos.azure.com:443'
DATABASE = 'osint-db'


def clear_all_data(client: Client, graph: str):
    """Clear all vertices and edges from the graph."""
    print(f"Clearing all data from graph '{graph}'...")
    
    try:
        # Get initial stats
        initial_vertices, initial_edges = get_graph_stats(client)
        if initial_vertices is not None and initial_edges is not None:
            print(f"Initial state: {initial_vertices} vertices, {initial_edges} edges")
        
        # Clear edges in batches to avoid rate limits
        print("Clearing edges in batches...")
        batch_size = 1000
        edges_cleared = 0
        
        while True:
            try:
                # Get count of remaining edges
                remaining_edges = client.submit("g.E().count()").all().result()[0]
                if remaining_edges == 0:
                    break
                
                # Clear a batch of edges
                result = client.submit(f"g.E().limit({batch_size}).drop()").all().result()
                edges_cleared += batch_size
                print(f"Cleared {edges_cleared} edges... ({remaining_edges} remaining)")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                if "429" in str(e) or "TooManyRequests" in str(e):
                    print("Rate limit hit, waiting 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    raise
        
        # Clear vertices in batches
        print("Clearing vertices in batches...")
        vertices_cleared = 0
        
        while True:
            try:
                # Get count of remaining vertices
                remaining_vertices = client.submit("g.V().count()").all().result()[0]
                if remaining_vertices == 0:
                    break
                
                # Clear a batch of vertices
                result = client.submit(f"g.V().limit({batch_size}).drop()").all().result()
                vertices_cleared += batch_size
                print(f"Cleared {vertices_cleared} vertices... ({remaining_vertices} remaining)")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                if "429" in str(e) or "TooManyRequests" in str(e):
                    print("Rate limit hit, waiting 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    raise
        
        # Verify cleanup
        final_vertices, final_edges = get_graph_stats(client)
        if final_vertices is not None and final_edges is not None:
            print(f"Final state: {final_vertices} vertices, {final_edges} edges")
            print("✅ All data cleared successfully!")
        else:
            print("⚠️ Could not verify cleanup completion")
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        raise

def clear_partition_data(client: Client, graph: str, partition_key: str, partition_value: str):
    """Clear data for a specific partition."""
    print(f"Clearing data for partition '{partition_key}' = '{partition_value}' in graph '{graph}'...")
    
    try:
        # Get initial stats for this partition
        partition_vertices = client.submit(f"g.V().has('{partition_key}', '{partition_value}').count()").all().result()[0]
        print(f"Found {partition_vertices} vertices in partition '{partition_value}'")
        
        if partition_vertices == 0:
            print("No data found in this partition.")
            return
        
        # Clear edges first
        print("Clearing edges...")
        client.submit(f"g.V().has('{partition_key}', '{partition_value}').outE().drop()").all().result()
        client.submit(f"g.V().has('{partition_key}', '{partition_value}').inE().drop()").all().result()
        
        # Clear vertices
        print("Clearing vertices...")
        client.submit(f"g.V().has('{partition_key}', '{partition_value}').drop()").all().result()
        
        # Verify cleanup
        final_vertices = client.submit(f"g.V().has('{partition_key}', '{partition_value}').count()").all().result()[0]
        print(f"Final state: {final_vertices} vertices in partition '{partition_value}'")
        print("✅ Partition data cleared successfully!")
        
    except Exception as e:
        print(f"❌ Error during partition cleanup: {e}")
        raise

def list_partitions(client: Client, partition_key: str):
    """List all partition values in the graph."""
    try:
        print(f"Listing all partitions for key '{partition_key}'...")
        partitions = client.submit(f"g.V().values('{partition_key}').dedup()").all().result()
        print(f"Found {len(partitions)} partitions:")
        for i, partition in enumerate(sorted(partitions), 1):
            print(f"  {i:3d}. {partition}")
        return partitions
    except Exception as e:
        print(f"❌ Error listing partitions: {e}")
        return []

def main():
    """Command line interface for database cleanup."""
    ap = argparse.ArgumentParser(description="Clean up Cosmos DB Gremlin graph data")
    ap.add_argument("--graph", required=True, help="Name of the graph/container")
    ap.add_argument("--partition-key", help="Partition key field name (for partition-specific cleanup)")
    ap.add_argument("--partition-value", help="Specific partition value to clear")
    ap.add_argument("--list-partitions", action="store_true", help="List all partition values")
    ap.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    
    args = ap.parse_args()
    
    # Build client
    client = build_client(args.graph)
    
    try:
        # Show current stats
        vertex_count, edge_count = get_graph_stats(client)
        if vertex_count is not None and edge_count is not None:
            print(f"Current graph contains {vertex_count} nodes and {edge_count} relationships")
        
        # List partitions if requested
        if args.list_partitions:
            if not args.partition_key:
                print("❌ --partition-key is required when using --list-partitions")
                return
            list_partitions(client, args.partition_key)
            return
        
        # Confirmation prompt
        if not args.confirm:
            if args.partition_value:
                confirm_msg = f"Are you sure you want to clear all data for partition '{args.partition_value}'? (y/n): "
            else:
                confirm_msg = f"Are you sure you want to clear ALL data from graph '{args.graph}'? (y/n): "
            
            response = input(confirm_msg)
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return
        
        # Perform cleanup
        if args.partition_value and args.partition_key:
            clear_partition_data(client, args.graph, args.partition_key, args.partition_value)
        else:
            clear_all_data(client, args.graph)
            
    finally:
        # Always close the client connection
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Error closing client connection: {e}")

if __name__ == "__main__":
    main()
