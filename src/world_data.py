import pandas as pd
import numpy as np
import wbgapi as wb
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_rest_of_world(year: int, indicators, countries, unique_countries, external):
    try:
        # Retrieve full list of economies available from the API
        all_countries = {item['id'] for item in wb.economy.list()}

        # Identify all countries already used in this graph
        modelled_countries = set([c for c in unique_countries if c != 'ROW'])

        # Take complement set = Rest of the World
        rest_countries = list(all_countries - modelled_countries)

        if len(rest_countries) == 0:
            logger.warning("No remaining countries found for ROW computation")
        else:
            # Compute unweighted mean across remaining countries for each indicator
            row_values = {}
            for wb_code, col_name in indicators.items():
                try:
                    df_rest = wb.data.DataFrame(wb_code, rest_countries, time=year)
                    if str(year) in df_rest.columns:
                        series = df_rest[str(year)]
                    elif year in df_rest.columns:
                        series = df_rest[year]
                    elif len(df_rest.columns) == 1:
                        series = df_rest.iloc[:, 0]
                    else:
                        year_cols = [c for c in df_rest.columns if str(year) in str(c)]
                        series = df_rest[year_cols[0]] if year_cols else pd.Series(dtype=float)

                    mean_val = pd.to_numeric(series, errors='coerce').mean()
                    row_values[col_name] = mean_val
                    logger.info(f"ROW mean for {col_name}: {mean_val:.3f}")

                except Exception as e:
                    logger.warning(f"    ROW {col_name}: failed ({type(e).__name__}) - {e}")
                    row_values[col_name] = np.nan

            # Assign those averages to all ROW_* nodes
            row_nodes = countries[countries == 'ROW'].index
            for col_name, mean_val in row_values.items():
                external.loc[row_nodes, col_name] = mean_val

            logger.info(f"Filled {len(row_nodes)} ROW nodes with rest-of-world average indicator values")
    except Exception as e:
        logger.error(f"Failed to compute ROW averages: {type(e).__name__} - {e}")
    
region_map_override = {
    'TWN': 'EAS',  # Taiwan -> East Asia & Pacific
}

def fill_with_region_mean(external, indicators, year):
    """
    Fill missing country-year indicator values using World Bank region aggregates.
    """
    for wb_code, col_name in indicators.items():
        missing_countries = external[external[col_name].isna()].index

        filled = 0
        for node in missing_countries:
            # Get the region code
            try:
                c = node.split('_')[0]
                if c == "ROW": # This should not happen
                    logger.info("Found 'ROW', skipping.")
                    continue

                if c in region_map_override:
                    region = region_map_override[c]
                else:
                    meta = wb.economy.get(c)
                    region = meta['region']

                if not region:
                    logger.warning(f"No region info for {c}; skipping regional fill")
                    continue

                # Query region-level data
                df_region = wb.data.DataFrame(wb_code, region, time=year)

                if str(year) in df_region.columns:
                    value = pd.to_numeric(df_region[str(year)], errors='coerce').mean()
                elif len(df_region.columns) > 0:
                    value = pd.to_numeric(df_region.iloc[:, 0], errors='coerce').mean()
                else:
                    value = np.nan

                if pd.isna(value):
                    df_world = wb.data.DataFrame(wb_code, 'WLD', time=year)
                    if str(year) in df_world.columns:
                        value = pd.to_numeric(df_world[str(year)], errors='coerce').mean()
                        logger.info(f"  {c} {col_name}: Regional mean not available, filled with WORLD mean ({value:.3f})")
                    elif len(df_world.columns) > 0:
                        value = pd.to_numeric(df_world.iloc[:, 0], errors='coerce').mean()
                        logger.info(f"  {c} {col_name}: Regional mean not available, filled with WORLD mean ({value:.3f})")
                    else:
                        value = np.nan
                        logger.info(f"  {c} {col_name}: Regional mean and world main not available. Value set to NaN.")

                if pd.notna(value):
                    external.loc[node, col_name] = value
                    filled += 1

            except Exception as e:
                logger.warning(f"Failed region fill for {c}, {col_name}: {type(e).__name__} - {e}")
            
        logger.info(f"Filled {filled}/{len(missing_countries)} missing nodes with regional average indicator values")

    return external

def load_indicators(year: int, nodes: pd.Index) -> pd.DataFrame:
    """
    Load selective external data for given year.
    Only includes high-impact economic indicators for supply chain analysis.
    
    Priority indicators (6 features):
    1. GDP per capita (log) - Economic capacity
    2. GDP growth - Economic momentum  
    3. Exports % GDP - Trade openness (exports)
    4. Imports % GDP - Import dependency
    5. Trade % GDP - Overall trade exposure
    6. Inflation - Economic instability
    
    Args:
        year: Year to load data for
        nodes: Index of node labels (e.g., ['USA_MFG', 'CHN_SVC', ...])
    
    Returns:
        DataFrame indexed by node labels with country-level features broadcasted
        to all nodes from the same country.
    """
    try:
        # Extract country codes from node labels (format: COUNTRY_SECTOR)
        countries = pd.Series([n.split('_')[0] for n in nodes], index=nodes)
        unique_countries = countries.unique()
        
        logger.info(f"Loading indicators for {year} - {len(unique_countries)} countries")
        
        # World Bank indicator codes
        indicators = {
            'NY.GDP.PCAP.CD': 'gdp_per_capita',
            'NY.GDP.MKTP.KD.ZG': 'gdp_growth', 
            'NE.EXP.GNFS.ZS': 'exports_pct_gdp',
            'NE.IMP.GNFS.ZS': 'imports_pct_gdp',
            'NE.TRD.GNFS.ZS': 'trade_pct_gdp',
            'NY.GDP.DEFL.KD.ZG': 'inflation',
            'SL.UEM.TOTL.ZS': 'unemployment_rate',
            # Not enough data for account balance, tariffs, etc.
        }
        
        external = pd.DataFrame(index=nodes)
        success_count = 0
        
        for wb_code, col_name in indicators.items():
            try:
                # Fetch country data: returns DataFrame with country codes as index
                df = wb.data.DataFrame(wb_code, unique_countries, time=year)

                # Handle different return formats from World Bank API
                if str(year) in df.columns:
                    country_series = df[str(year)]
                elif year in df.columns:
                    country_series = df[year]
                elif len(df.columns) == 1:
                    country_series = df.iloc[:, 0]
                else:
                    # Try to find the year column by matching
                    year_cols = [c for c in df.columns if str(year) in str(c)]
                    if year_cols:
                        country_series = df[year_cols[0]]
                    else:
                        raise KeyError(f"Could not find year {year} in columns: {df.columns.tolist()}")
                
                # Map directly: countries maps node->country, then map country->value
                external[col_name] = countries.map(country_series)
                success_count += 1
                
            except KeyError as e:
                logger.error(f"  ✗ {col_name} ({wb_code}): Column key error - {e}")
                logger.error(f"    Available columns: {df.columns.tolist() if 'df' in locals() else 'N/A'}")
                external[col_name] = np.nan
                
            except Exception as e:
                logger.error(f"  ✗ {col_name} ({wb_code}): {type(e).__name__} - {e}")
                external[col_name] = np.nan
        
        logger.info(f"Successfully loaded {success_count}/{len(indicators)} indicators")
        
        # Handle ROW (Rest Of World)
        if 'ROW' in countries.values:
            logger.info("Detected 'ROW' nodes — computing Rest of World averages")
            calculate_rest_of_world(year, indicators, countries, unique_countries, external)

        # Fill missing country data with regional or world mean
        external = fill_with_region_mean(external, indicators, year)

        # Log-transform highly skewed GDP per capita
        if 'gdp_per_capita' in external.columns:
            external['log_gdp_per_capita'] = np.log1p(external['gdp_per_capita'])
            external = external.drop('gdp_per_capita', axis=1)
        
        return external
        
    except Exception as e:
        logger.error(f"Fatal error loading external data for {year}: {type(e).__name__} - {e}")
        logger.error(f"Returning empty DataFrame")
        # Return empty DataFrame with same index if loading fails
        return pd.DataFrame(index=nodes)


def test_world_bank_connection():
    """Test function to diagnose World Bank API issues"""
    logger.info("Testing World Bank API connection...")
    
    try:
        # Test basic connectivity
        test_year = 1995
        test_countries = ['USA', 'CHN', 'DEU']
        test_indicator = 'NY.GDP.PCAP.CD'
        
        logger.info(f"Fetching {test_indicator} for {test_countries} in {test_year}")
        df = wb.data.DataFrame(test_indicator, test_countries, time=test_year)
        
        logger.info(f"Success! Returned data:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Index: {df.index.tolist()}")
        logger.info(f"  Data:\n{df}")
        
        return True
        
    except Exception as e:
        logger.error(f"World Bank API test failed: {type(e).__name__} - {e}")
        return False


if __name__ == "__main__":
    # Run diagnostic test
    test_world_bank_connection()