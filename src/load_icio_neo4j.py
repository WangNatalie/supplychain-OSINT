import pandas as pd
import numpy as np
from neo4j import GraphDatabase

# ---- Neo4j connection ----
URI = "neo4j+s://c7b4c3b9.databases.neo4j.io" 
USER = "neo4j"
PASSWORD = "XLvZoNtC0F9aArkiTryipJTGeldkEKtkcIcQ_7-Tktk"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ---- Load ICIO CSV ----
df = pd.read_csv("../ICIO/2022_SML.csv")

# Get row and column identifiers
row_ids = df["V1"]
col_ids = df.columns[1:]  # skip the V1 column

# ---- Filter out non-intermediate use columns ----
# Final demand categories to exclude
final_demand_codes = ['HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR', 'OUT']

# Keep only columns that represent intermediate use (industry-to-industry flows)
sector_cols = []
num_skips = 0
print("First ten final use columns skipped: ")
for col in col_ids:
    col_str = str(col)
    # Skip if it's a final demand category
    if any(fd_code in col_str for fd_code in final_demand_codes):
        if num_skips < 10:
            print(col_str)
            num_skips+=1
        if 'OUT' in col_str:
            print(f"Skipping {col_str} column")
        continue
    # Skip if it doesn't follow the COUNTRY_INDUSTRY pattern
    if '_' not in col_str:
        continue
    sector_cols.append(col)

print(f"Found {len(sector_cols)} intermediate use columns out of {len(col_ids)} total columns")

# ---- Filter out non-intermediate use rows ----
# Categories to exclude from rows (exact word matches only)
exclude_row_patterns = ['TLS', 'VA', 'OUT']

# Keep only rows that represent producing sectors
sector_rows = []
for i, row_id in enumerate(row_ids):
    row_str = str(row_id)
    # Skip if it exactly matches excluded patterns
    if row_str in exclude_row_patterns:
        continue
    # Skip if it doesn't follow the COUNTRY_INDUSTRY pattern
    if '_' not in row_str:
        continue
    sector_rows.append((i, row_id))

print(f"Found {len(sector_rows)} intermediate use rows out of {len(row_ids)} total rows")

# Show ALL excluded rows
print("\nAll excluded rows:")
excluded_row_indices = [i for i in range(len(row_ids)) if i not in [idx for idx, _ in sector_rows]]
excluded_rows = [row_ids[i] for i in excluded_row_indices]
for row in excluded_rows:
    print(f"  {row}")

print(f"\n{'='*50}")
print("Collecting intermediate matrix values for statistical analysis...")

# ---- Collect all intermediate matrix values ----
intermediate_values = []
for row_idx, src in sector_rows:
    for tgt in sector_cols:
        col_idx = df.columns.get_loc(tgt)
        value = df.iloc[row_idx, col_idx]
        
        # Only collect non-zero, non-null values
        if pd.notna(value) and value > 0:
            intermediate_values.append(float(value))

# Convert to numpy array for statistical calculations
values_array = np.array(intermediate_values)

print(f"Collected {len(intermediate_values)} non-zero intermediate values")

# ---- Calculate statistics ----
median_value = np.median(values_array)
std_value = np.std(values_array)
threshold_value = np.percentile(values_array, 99)

# Calculate percentiles
percentiles = np.percentile(values_array, [5, 10, 25, 50, 75, 90, 95, 99])

print(f"\nStatistical Summary:")
print(f"{'='*40}")
print(f"Median value: ${median_value:.2f}M")
print(f"Standard deviation: ${std_value:.2f}M")
print(f"99th percentile threshold: ${threshold_value:.2f}M")
print(f"Min value: ${np.min(values_array):.2f}M")
print(f"Max value: ${np.max(values_array):.2f}M")

print(f"\nPercentile Distribution:")
print(f"{'='*40}")
percentile_labels = ['5th', '10th', '25th', '50th (Median)', '75th', '90th', '95th', '99th']
for label, perc in zip(percentile_labels, percentiles):
    print(f"{label:15}: ${perc:>10.2f}M")

# Count values below threshold
values_below_threshold = np.sum(values_array < threshold_value)
values_above_threshold = len(values_array) - values_below_threshold
print(f"\nFiltering Summary:")
print(f"{'='*40}")
print(f"Values below threshold: {values_below_threshold:,} ({values_below_threshold/len(values_array)*100:.1f}%)")
print(f"Values above threshold: {values_above_threshold:,} ({values_above_threshold/len(values_array)*100:.1f}%)")

print(f"\n{'='*50}")
print("Starting graph creation...")

def create_edge(tx, src, tgt, value):
    """Create a new edge between two sectors."""
    try:
        src_country, src_sector = src.split("_", 1)
        tgt_country, tgt_sector = tgt.split("_", 1)
    except ValueError:
        print(f"Warning: Could not parse {src} or {tgt}")
        return False

    query = """
    MERGE (s:Sector {country: $src_country, code: $src_sector})
    MERGE (t:Sector {country: $tgt_country, code: $tgt_sector})
    CREATE (s)-[r:FLOW]->(t)
    SET r.value = $value
    """
    tx.run(query,
           src_country=src_country, src_sector=src_sector,
           tgt_country=tgt_country, tgt_sector=tgt_sector,
           value=value)
    return True


def upsert_edge(tx, src, tgt, value):
    """Create or update edge in one operation using MERGE."""
    try:
        src_country, src_sector = src.split("_", 1)
        tgt_country, tgt_sector = tgt.split("_", 1)
    except ValueError:
        print(f"Warning: Could not parse {src} or {tgt}")
        return False

    query = """
    MERGE (s:Sector {country: $src_country, code: $src_sector})
    MERGE (t:Sector {country: $tgt_country, code: $tgt_sector})
    MERGE (s)-[r:FLOW]->(t)
    SET r.value = $value
    """
    tx.run(query,
           src_country=src_country, src_sector=src_sector,
           tgt_country=tgt_country, tgt_sector=tgt_sector,
           value=value)
    return True

# ---- Process the intermediate use matrix with statistical filtering ----
with driver.session() as session:
    edges_processed = 0
    edges_skipped = 0
    
    for row_idx, src in sector_rows:
        for tgt in sector_cols:
            col_idx = df.columns.get_loc(tgt)
            value = df.iloc[row_idx, col_idx]
            
            # Only process values above threshold
            if pd.notna(value) and value >= threshold_value:
                try:
                    session.execute_write(upsert_edge, src, tgt, float(value))
                    edges_processed += 1
                    
                    if edges_processed % 1000 == 0:
                        print(f"Processed {edges_processed} edges above threshold...")
                        
                except Exception as e:
                    print(f"Error processing {src} -> {tgt}: {e}")
            elif pd.notna(value) and value > 0:
                edges_skipped += 1

print(f"\nFinished indexing ICIO table:")
print(f"{'='*40}")
print(f"Processed {edges_processed:,} new edges above threshold (${threshold_value:.2f}M)")
print(f"Skipped {edges_skipped:,} edges below threshold")

driver.close()