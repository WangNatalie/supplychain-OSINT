import requests
import pandas as pd
import sys

# Determine mode (exports or imports)
mode = sys.argv[1] if len(sys.argv) > 1 else "exports"
if mode not in ["exports", "imports"]:
    print("Usage: python query_ports.py [exports|imports]")
    print("Default: exports")
    mode = "exports"

# Set URL based on mode
if mode == "exports":
    url = "https://api.census.gov/data/timeseries/intltrade/exports/porths"
    prefix = "E"
    value_var = "ALL_VAL_MO"
else:  # imports
    url = "https://api.census.gov/data/timeseries/intltrade/imports/porths"
    prefix = "I"
    value_var = "GEN_VAL_MO"

params = {
    "get": f"{value_var},{prefix}_COMMODITY,{prefix}_COMMODITY_SDESC,CTY_CODE,MONTH",
    "PORT": "2704",  # Port of Long Beach
    "YEAR": "2021",
}

response = requests.get(url, params=params)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=data[0])

# Clean numeric columns and remove commodities with 0 import/export value, unidentified HS/CTY codes
df[f"{value_var}"] = pd.to_numeric(df[f"{value_var}"], errors="coerce")
df = df[df[f"{value_var}"] > 0]
df = df[df[f"{prefix}_COMMODITY"] != "-"]
df = df[df["CTY_CODE"] != "-"]

# Filter for August–October
df_filtered = df[df["MONTH"].isin(["08", "09", "10"])]

# Save filtered data
df_filtered.to_csv(f"long_beach_{mode}_AugOct2021.csv", index=False)

# --- Pivot so each commodity has columns for Aug, Sep, Oct ---
pivot = df_filtered.pivot_table(
    index=[f"{prefix}_COMMODITY", f"{prefix}_COMMODITY_SDESC", "CTY_CODE"],
    columns="MONTH",
    values=f"{value_var}",
    aggfunc="sum"
).reset_index()

# Rename columns for convenience
pivot = pivot.rename(columns={"08": "Aug", "09": "Sep", "10": "Oct"})

# --- Find commodities with minimum in September ---
mask = (pivot["Aug"] > pivot["Sep"]) & (pivot["Oct"] > pivot["Sep"])
min_in_sept = pivot[mask]

# --- 4. Calculate total trade values ---
total_all_val = pivot[["Aug", "Sep", "Oct"]].sum().sum()
min_in_sept_val = min_in_sept[["Aug", "Sep", "Oct"]].sum().sum()
percentage = (min_in_sept_val / total_all_val * 100) if total_all_val > 0 else 0

# --- 5. Sum total trade for all commodities by month ---
all_totals_by_month = pivot[["Aug", "Sep", "Oct"]].sum()
all_totals_by_month_df = pd.DataFrame(all_totals_by_month).reset_index()
all_totals_by_month_df.columns = ["Month", f"Total_{value_var}"]

# --- 6. Output ---
print(f"Mode: {mode.upper()}")
print(f"Commodities with minimum in September: {len(min_in_sept)}")
print(f"Share of total {mode} value (Aug–Oct): {percentage:.2f}%\n")
print(f"Total {mode} values (USD) for all commodities:")
print(all_totals_by_month_df)

# --- 7. Save list of commodities with minimum in Sept 2021 to CSV ---
min_in_sept.to_csv(f"commodities_min_in_september_{mode}.csv", index=False)


