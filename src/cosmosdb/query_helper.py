#!/usr/bin/env python3
"""
query_helper.py

Test script to query flow values between specific country-sector combinations
in the Cosmos DB Gremlin graph.

Usage:
python query_helper.py --graph "<graph_name>" --from "USA_MANUF" --to "CHN_AGRI"
python query_helper.py --graph "<graph_name>" --from "USA_MANUF" --to "CHN_AGRI" --list-connections
"""

import argparse
import time
from datetime import datetime
from gremlin_python.driver.client import Client
from gremlin_python.driver.protocol import GremlinServerError
from dotenv import load_dotenv
from check_db_stats import build_client, get_graph_stats

load_dotenv()

# Cosmos DB endpoint
SQL_ENDPOINT = 'https://graph-db-osint.documents.azure.com:443/'
GREMLIN_ENDPOINT = 'wss://graph-db-osint.gremlin.cosmos.azure.com:443'
DATABASE = 'osint-db'


def time_operation(operation_name):
    """Decorator to time operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"⏱️  Starting {operation_name}...")
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                print(f"✅ {operation_name} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                print(f"❌ {operation_name} failed after {duration:.2f} seconds: {e}")
                raise
        return wrapper
    return decorator


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


def check_node_exists(client: Client, node_id: str):
    """
    Check if a node exists in the graph.
    
    Args:
        client: Gremlin client
        node_id: Node ID to check
    
    Returns:
        bool: True if node exists, False otherwise
    """
    try:
        # Use count() instead of hasNext() since hasNext() is not supported in Cosmos DB
        query = f"g.V('{node_id}').count()"
        result = client.submit(query).all().result()
        return result[0] > 0 if result else False
    except Exception as e:
        print(f"Error checking if node '{node_id}' exists: {e}")
        return False


def get_node_properties(client: Client, node_id: str):
    """
    Get properties of a specific node.
    
    Args:
        client: Gremlin client
        node_id: Node ID to query
    
    Returns:
        dict: Node properties, or None if node doesn't exist
    """
    try:
        query = f"g.V('{node_id}').valueMap(true)"
        result = client.submit(query).all().result()
        
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        print(f"Error getting properties for node '{node_id}': {e}")
        return None


def get_random_node(client: Client):
    """
    Get a random node and its properties from the graph.
    
    Args:
        client: Gremlin client
    
    Returns:
        tuple: (node_id, properties) or (None, None) if no nodes exist
    """
    try:
        # First get a random node ID
        query = "g.V().sample(1).id()"
        result = client.submit(query).all().result()
        
        if not result:
            return None, None
        
        node_id = result[0]
        
        # Get properties of the random node
        properties = get_node_properties(client, node_id)
        
        return node_id, properties
        
    except Exception as e:
        print(f"Error getting random node: {e}")
        return None, None


def find_first_edge(client: Client):
    """
    Get the first edge and its properties from the graph.
    
    Args:
        client: Gremlin client
    
    Returns:
        dict: Edge information with source, target, and properties, or None if no edges exist
    """
    try:
        # Get the first edge and project its properties and endpoint ids
        query = (
            "g.E().limit(1).as('e')"
            ".project('e','source','target')"
            ".by(valueMap(true))"
            ".by(outV().id())"
            ".by(inV().id())"
        )
        result = client.submit(query).all().result()
        
        if not result:
            return None
        
        edge_data = result[0]
        
        # Extract edge properties
        edge_props = edge_data.get('e', {})
        source_id = edge_data.get('source', '')
        target_id = edge_data.get('target', '')
        
        return {
            'source': source_id,
            'target': target_id,
            'properties': edge_props
        }
        
    except Exception as e:
        print(f"Error getting first edge: {e}")
        return None


def get_flow_value(client: Client, from_sector: str, to_sector: str):
    """
    Get the flow value from one country-sector to another.
    
    Args:
        client: Gremlin client
        from_sector: Source country-sector (e.g., "USA_MANUF")
        to_sector: Target country-sector (e.g., "CHN_AGRI")
    
    Returns:
        float: Flow value, or None if no connection exists
    """
    try:
        # First check if both nodes exist
        if not check_node_exists(client, from_sector):
            print(f"Source node '{from_sector}' does not exist")
            return None
            
        if not check_node_exists(client, to_sector):
            print(f"Target node '{to_sector}' does not exist")
            return None
        
        # Query for the edge between the two sectors using a simpler approach
        # First get all outgoing edges from the source
        query = f"g.V('{from_sector}').outE('million_USD').as('e').inV().hasId('{to_sector}').select('e').values('value')"
        result = client.submit(query).all().result()
        
        if result:
            return float(result[0])
        else:
            return None
            
    except Exception as e:
        print(f"Error querying flow value: {e}")
        return None


def get_all_outgoing_flows(client: Client, from_sector: str, min_value: float = 0.0):
    """
    Get all outgoing flows from a specific country-sector.
    
    Args:
        client: Gremlin client
        from_sector: Source country-sector
        min_value: Minimum flow value to include
    
    Returns:
        list: List of tuples (target_sector, flow_value)
    """
    try:
        # First check if the source node exists
        if not check_node_exists(client, from_sector):
            print(f"Source node '{from_sector}' does not exist")
            return []
        
        query = f"g.V('{from_sector}').outE('million_USD').as('e').inV().as('v').select('e', 'v').by('value').by('id')"
        result = client.submit(query).all().result()
        
        flows = []
        for item in result:
            if isinstance(item, dict) and 'e' in item and 'v' in item:
                flow_value = float(item['e'])
                target_sector = item['v']
                if flow_value >= min_value:
                    flows.append((target_sector, flow_value))
        
        return sorted(flows, key=lambda x: x[1], reverse=True)  # Sort by value descending
        
    except Exception as e:
        print(f"Error querying outgoing flows: {e}")
        return []


@time_operation("Disruption Simulation")
def simulate_disruption(client: Client, disrupted_node: str, reduction_percentage: float, max_degrees: int = 3, min_flow_value: float = 0.0):
    """
    Simulate a disruption by reducing a node's output capacity and analyze cascading effects.
    
    Args:
        client: Gremlin client
        disrupted_node: Node ID to disrupt
        reduction_percentage: Percentage reduction (0.0 to 1.0, e.g., 0.2 for 20% reduction)
        max_degrees: Maximum degrees of separation to analyze (default 3)
        min_flow_value: Minimum flow value to include in analysis (default 0.0)
    
    Returns:
        dict: Analysis results with affected nodes and flows by degree
    """
    try:
        # Check if the disrupted node exists
        if not check_node_exists(client, disrupted_node):
            print(f"Disrupted node '{disrupted_node}' does not exist")
            return None
        
        print(f"Simulating {reduction_percentage*100:.1f}% capacity reduction for '{disrupted_node}'")
        print(f"Analyzing cascading effects up to {max_degrees} degrees...")
        print(f"Minimum flow threshold: ${min_flow_value:,.2f} million")
        print("Using streaming approach to avoid rate limits...")
        
        # Get original outgoing flows from the disrupted node
        original_flows = get_all_outgoing_flows(client, disrupted_node, min_flow_value)
        if not original_flows:
            print(f"No outgoing flows found from '{disrupted_node}'")
            return None
        
        print(f"\nDirect effects (Degree 1) - {len(original_flows)} affected flows:")
        affected_flows = {}
        affected_flows[1] = []
        
        for target, original_value in original_flows:
            new_value = original_value * (1 - reduction_percentage)
            reduction_amount = original_value - new_value
            affected_flows[1].append({
                'source': disrupted_node,
                'target': target,
                'original_value': original_value,
                'new_value': new_value,
                'reduction': reduction_amount,
                'reduction_pct': reduction_percentage
            })
            print(f"  {original_value:>12,.6f} → {new_value:>12,.6f} ({reduction_amount:>12,.6f}) → {target}")
        
        # Analyze cascading effects for subsequent degrees with rate limiting
        for degree in range(2, max_degrees + 1):
            print(f"\nCascading effects (Degree {degree}):")
            affected_flows[degree] = []
            
            # Get nodes affected in the previous degree
            prev_degree_targets = [flow['target'] for flow in affected_flows[degree - 1]]
            
            # Process nodes in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(prev_degree_targets), batch_size):
                batch = prev_degree_targets[i:i + batch_size]
                print(f"  Processing batch {i//batch_size + 1}/{(len(prev_degree_targets) + batch_size - 1)//batch_size}...")
                
                for affected_node in batch:
                    try:
                        # Add delay to avoid rate limits
                        time.sleep(0.1)
                        
                        # Check if this node has outgoing flows
                        node_flows = get_all_outgoing_flows(client, affected_node, min_flow_value)
                        if not node_flows:
                            continue
                        
                        # Calculate proportional reduction based on input reduction
                        # Find how much this node's input was reduced
                        input_reduction = 0
                        for prev_flow in affected_flows[degree - 1]:
                            if prev_flow['target'] == affected_node:
                                input_reduction = prev_flow['reduction_pct']
                                break
                        
                        # Apply proportional reduction to outputs
                        for target, original_value in node_flows:
                            # Skip if we've already processed this flow
                            if any(flow['source'] == affected_node and flow['target'] == target 
                                  for flows in affected_flows.values() for flow in flows):
                                continue
                            
                            new_value = original_value * (1 - input_reduction)
                            reduction_amount = original_value - new_value
                            
                            affected_flows[degree].append({
                                'source': affected_node,
                                'target': target,
                                'original_value': original_value,
                                'new_value': new_value,
                                'reduction': reduction_amount,
                                'reduction_pct': input_reduction
                            })
                            print(f"    {original_value:>12,.6f} → {new_value:>12,.6f} ({reduction_amount:>12,.6f}) → {target}")
                    
                    except Exception as e:
                        print(f"    Error processing node '{affected_node}': {e}")
                        continue
                
                # Longer delay between batches
                if i + batch_size < len(prev_degree_targets):
                    print(f"  Waiting 2 seconds before next batch...")
                    time.sleep(2)
        
        # Calculate summary statistics
        total_original_value = sum(flow['original_value'] for flows in affected_flows.values() for flow in flows)
        total_new_value = sum(flow['new_value'] for flows in affected_flows.values() for flow in flows)
        total_reduction = total_original_value - total_new_value
        
        print(f"\nSummary:")
        print(f"  Total flows analyzed: {sum(len(flows) for flows in affected_flows.values())}")
        print(f"  Total original value: ${total_original_value:,.2f} million")
        print(f"  Total new value: ${total_new_value:,.2f} million")
        print(f"  Total economic impact: ${total_reduction:,.2f} million")
        print(f"  Overall reduction: {(total_reduction/total_original_value)*100:.2f}%")
        
        return {
            'disrupted_node': disrupted_node,
            'reduction_percentage': reduction_percentage,
            'max_degrees': max_degrees,
            'affected_flows': affected_flows,
            'summary': {
                'total_flows': sum(len(flows) for flows in affected_flows.values()),
                'total_original_value': total_original_value,
                'total_new_value': total_new_value,
                'total_reduction': total_reduction,
                'overall_reduction_pct': (total_reduction/total_original_value)*100
            }
        }
        
    except Exception as e:
        print(f"Error during disruption simulation: {e}")
        return None

def get_all_incoming_flows(client: Client, to_sector: str, min_value: float = 0.0):
    """
    Get all incoming flows to a specific country-sector.
    
    Args:
        client: Gremlin client
        to_sector: Target country-sector
        min_value: Minimum flow value to include
    
    Returns:
        list: List of tuples (source_sector, flow_value)
    """
    try:
        # First check if the target node exists
        if not check_node_exists(client, to_sector):
            print(f"Target node '{to_sector}' does not exist")
            return []
        
        query = f"g.V('{to_sector}').inE('million_USD').as('e').outV().as('v').select('e', 'v').by('value').by('id')"
        result = client.submit(query).all().result()
        
        flows = []
        for item in result:
            if isinstance(item, dict) and 'e' in item and 'v' in item:
                flow_value = float(item['e'])
                source_sector = item['v']
                if flow_value >= min_value:
                    flows.append((source_sector, flow_value))
        
        return sorted(flows, key=lambda x: x[1], reverse=True)  # Sort by value descending
        
    except Exception as e:
        print(f"Error querying incoming flows: {e}")
        return []


@time_operation("Query Operations")
def test_query_helper(graph: str, from_sector: str = None, to_sector: str = None, 
                   list_connections: bool = False, min_value: float = 0.0):
    """
    Test flow queries on the graph.
    
    Args:
        graph: Graph name
        from_sector: Source country-sector
        to_sector: Target country-sector
        list_connections: Whether to list all connections for the given sector
        min_value: Minimum flow value to display
    """
    print(f"Connecting to graph '{graph}'...")
    
    # Build client
    client = build_client(graph)
    
    try:
        # Get initial stats
        vertex_count, edge_count = get_graph_stats(client)
        if vertex_count is not None and edge_count is not None:
            print(f"Graph contains {vertex_count} vertices and {edge_count} edges")
        
        if from_sector and to_sector:
            # Query specific flow
            print(f"\nQuerying flow from '{from_sector}' to '{to_sector}'...")
            flow_value = get_flow_value(client, from_sector, to_sector)
            
            if flow_value is not None:
                print(f"Flow value: {flow_value:,.6f}")
            else:
                print("No direct flow found between these sectors")
        
        if list_connections:
            if from_sector:
                # List all outgoing flows
                print(f"\nAll outgoing flows from '{from_sector}' (min value: {min_value}):")
                outgoing_flows = get_all_outgoing_flows(client, from_sector, min_value)
                
                if outgoing_flows:
                    print(f"Found {len(outgoing_flows)} outgoing flows (in millions USD):")
                    for i, (target, value) in enumerate(outgoing_flows[:20], 1):  # Show top 20
                        print(f"  {i:2d}. {value:>15,.6f} → {target:<20}")
                    if len(outgoing_flows) > 20:
                        print(f"  ... and {len(outgoing_flows) - 20} more")
                else:
                    print("No outgoing flows found")
            
            if to_sector:
                # List all incoming flows
                print(f"\nAll incoming flows to '{to_sector}' (min value: {min_value}):")
                incoming_flows = get_all_incoming_flows(client, to_sector, min_value)
                
                if incoming_flows:
                    print(f"Found {len(incoming_flows)} incoming flows (in millions USD):")
                    for i, (source, value) in enumerate(incoming_flows[:20], 1):  # Show top 20
                        print(f"  {i:2d}. {value:>15,.6f} → {source:<20}")
                    if len(incoming_flows) > 20:
                        print(f"  ... and {len(incoming_flows) - 20} more")
                else:
                    print("No incoming flows found")
            
    except Exception as e:
        print(f"Error during query: {e}")
        raise
    finally:
        # Always close the client connection
        try:
            client.close()
        except Exception as e:
            print(f"Warning: Error closing client connection: {e}")


def main():
    """Command line interface for testing flow queries."""
    script_start_time = time.time()
    print(f"🚀 Starting query_helper.py at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ap = argparse.ArgumentParser(description="Test flow queries on Cosmos DB Gremlin graph")
    ap.add_argument("--graph", required=True, help="Name of the graph/container")
    ap.add_argument("--from", dest="from_sector", help="Source country-sector (e.g., 'USA_MANUF')")
    ap.add_argument("--to", dest="to_sector", help="Target country-sector (e.g., 'CHN_AGRI')")
    ap.add_argument("--list-connections", action="store_true", 
                   help="List all connections for the specified sector(s)")
    ap.add_argument("--min-value", type=float, default=0.0,
                   help="Minimum flow value to display when listing connections")
    ap.add_argument("--check-node", help="Check if a specific node exists and show its properties")
    ap.add_argument("--random-node", action="store_true", help="Get a random node and show its properties")
    ap.add_argument("--first-edge", action="store_true", help="Get the first edge and show its properties")
    ap.add_argument("--simulate-disruption", help="Simulate disruption for a specific node")
    ap.add_argument("--reduction-pct", type=float, default=0.2, help="Reduction percentage for disruption (0.0-1.0)")
    ap.add_argument("--max-degrees", type=int, default=3, help="Maximum degrees of cascading effects to analyze")
    ap.add_argument("--min-disruption-flow", type=float, default=0.0, help="Minimum flow value to include in disruption analysis")
    
    args = ap.parse_args()
    
    # Validate arguments
    if not args.from_sector and not args.to_sector and not args.check_node and not args.random_node and not args.first_edge and not args.simulate_disruption:
        print("   No sectors specified. Use --from and/or --to to query specific flows.")
        print("   Use --list-connections to see all flows for a sector.")
        print("   Use --check-node to verify if a node exists.")
        print("   Use --random-node to get a random node.")
        print("   Use --first-edge to get the first edge.")
        print("   Use --simulate-disruption to analyze supply chain disruption effects.")
    
    # Handle node checking
    if args.check_node:
        print(f"Checking if node '{args.check_node}' exists...")
        client = build_client(args.graph)
        try:
            exists = check_node_exists(client, args.check_node)
            if exists:
                print(f"Node '{args.check_node}' exists")
                properties = get_node_properties(client, args.check_node)
                if properties:
                    print("Node properties:")
                    for key, value in properties.items():
                        if isinstance(value, list) and len(value) == 1:
                            print(f"  {key}: {value[0]}")
                        else:
                            print(f"  {key}: {value}")
            else:
                print(f"Node '{args.check_node}' does not exist")
        finally:
            client.close()
        return
    
    # Handle random node
    if args.random_node:
        print(f"Getting a random node...")
        client = build_client(args.graph)
        try:
            node_id, properties = get_random_node(client)
            if node_id and properties:
                print(f"Random node: {node_id}")
                print("Node properties:")
                for key, value in properties.items():
                    if isinstance(value, list) and len(value) == 1:
                        print(f"  {key}: {value[0]}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("No nodes found in the graph")
        finally:
            client.close()
        return
    
    # Handle first edge
    if args.first_edge:
        print("Getting the first edge...")
        client = build_client(args.graph)
        try:
            edge_data = find_first_edge(client)
            if edge_data:
                print(f"First edge: {edge_data['source']} → {edge_data['target']}")
                print("Edge properties:")
                for key, value in edge_data['properties'].items():
                    if isinstance(value, list) and len(value) == 1:
                        print(f"  {key}: {value[0]}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("No edges found in the graph")
        finally:
            client.close()
        return
    
    # Handle disruption simulation
    if args.simulate_disruption:
        print(f"Simulating disruption for node '{args.simulate_disruption}'...")
        client = build_client(args.graph)
        try:
            result = simulate_disruption(client, args.simulate_disruption, args.reduction_pct, args.max_degrees, args.min_disruption_flow)
            if result:
                print(f"\nDisruption simulation completed successfully!")
            else:
                print(f"\nDisruption simulation failed!")
        finally:
            client.close()
        return
    
    test_query_helper(
        args.graph,
        args.from_sector,
        args.to_sector,
        args.list_connections,
        args.min_value
    )
    
    # Print total execution time
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    print(f"\n🏁 Script completed in {total_duration:.2f} seconds")
    print(f"📊 Total execution time: {total_duration:.2f}s")


if __name__ == "__main__":
    main()
