#!/usr/bin/env python3
"""
build_shock_dataset.py - Build training dataset from shock events using UN Comtrade API

This script:
1. Loads shock event definitions
2. Identifies downstream dependencies from ICIO tables
3. Queries UN Comtrade API for import/export trade values (month-by-month)
4. Compares shock year vs baseline year
5. Stops when recovery detected or 12 months elapsed
6. Outputs training dataset with economic indicators and supplier metrics

Usage:
    python build_shock_dataset.py \
        --shocks shocks/shock_events.json \
        --icio-dir embeddings \
        --output shocks/training_data.csv \
        --top-k 10
"""

import json
import math
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from dotenv import load_dotenv
from ComtradeAPI import ComtradeAPI
from shock_helpers import (
    ICIOHelper,
    SupplierMetricsHelper,
    get_import_shock_series,
    month_keys,
    expected_from_pre_shock,
    baseline_year_for,
    slice_shock_and_baseline,
)

load_dotenv()

# Import ICIO utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ICIO.ICIO_parser import format_node_name
from world_data import load_indicators

class ShockDatasetBuilder:
    """Build training dataset from shock events"""
    
    def __init__(self, icio_dir: str, top_k: int = 10):
        """
        Args:
            icio_dir: Directory containing ICIO graph files
            top_k: Number of top downstream partners to track (FOREIGN only; see ICIOHelper.get_downstream_partners)
        """
        self.icio_dir = Path(icio_dir)
        self.top_k = top_k
        self.api = ComtradeAPI(rate_limit_delay=1.0)
        # Helpers (moved out of this file)
        self.icio = ICIOHelper(self.icio_dir)
        self.supplier_metrics = SupplierMetricsHelper(self.icio)
    
    def process_shock_event(self, shock: Dict) -> List[Dict]:
        """
        Process one shock event and build training samples
        
        Args:
            shock: Shock event definition
        
        Returns:
            List of training samples
        """
        print(f"\n{'='*80}")
        print(f"Processing shock: {shock['name']}")
        print(f"{'='*80}")
        
        shocked_node = shock['node']
        shock_date = shock['date']  # 'YYYY-MM'
        shock_year, shock_month = map(int, shock_date.split('-'))
        
        # Parse shocked country
        shocked_country = shocked_node.split('_')[0]
        shocked_sector = shocked_node.split('_')[1]  # Extract sector code (e.g., 'C29')
        
        print(f"Shocked node: {format_node_name(shocked_node)}")
        print(f"Date: {shock_date}")
        
        # Step 1: Get downstream partners from ICIO (FOREIGN only)
        print(f"\nStep 1: Identifying top {self.top_k} FOREIGN downstream partners from ICIO {shock_year}...")
        downstream = self.icio.get_downstream_partners(shocked_node, shock_year, self.top_k)
        
        print(f"Found {len(downstream)} downstream partners:")
        for i, partner in enumerate(downstream, 1):
            print(f"  {i}. {format_node_name(partner['target_node'])} - ${partner['edge_value']:,.0f}")
        
        baseline_year = baseline_year_for(shock_year)

        target_nodes = [partner['target_node'] for partner in downstream]
        
        # Specify only the 4 indicators we need
        needed_indicators = [
            'gdp_per_capita',      # Auto-transformed to log_gdp_per_capita
            'gdp_growth',
            'inflation',
            'unemployment_rate'
        ]
        
        print(
            f"\nStep 2: Loading {len(needed_indicators)} World Bank indicators for "
            f"{len(target_nodes)} downstream partners ({baseline_year})..."
        )
        try:
            indicators_df = load_indicators(baseline_year, pd.Index(target_nodes), indicator_list=needed_indicators)
            print(f"✓ Loaded {len(indicators_df.columns)} economic indicators")
        except Exception as e:
            print(f"  Failed to load economic indicators: {e}")
            indicators_df = None
        
        # Step 2: Query trade data for targets (imports for shock value, exports for propagation)
        print(f"\nStep 2: Querying UN Comtrade API for monthly trade data...")
        
        training_samples = []
        
        # Load graph once for all partners (used for weight calculations)
        shock_graph = self.icio.load_graph(shock_year)
        baseline_graph = self.icio.load_graph(baseline_year)

        # NOTE: `month_keys()` + `get_import_shock_series()` live in `shock_helpers.py`
        
        # Pre-shock window used for decomposition (same for all partners in this shock event)
        # With a 36-month query window, we have 24 months pre-shock + 12 months post-shock.
        pre_keys = month_keys(shock_year - 2, shock_month, 24)

        for partner in downstream:
            target_country = partner['target_country']
            target_node = partner['target_node']
            target_sector = partner['target_sector']
            
            print(f"\n  Partner: {format_node_name(target_node)}")
            # Domestic vs foreign ONLY affects how we source shock/baseline import series
            is_domestic = (target_country == shocked_country)
            print("    [DOMESTIC] Shock from ICIO" if is_domestic else f"    [FOREIGN] Shock from Comtrade imports ({target_country} from {shocked_country} {shocked_sector})")

            # Print supplier metrics + economic indicators ONCE per partner
            supplier_metrics = self.supplier_metrics.compute_supplier_metrics(target_node, shocked_node, shock_year)
            print(f"    Supplier HHI: {supplier_metrics['supplier_hhi']:.3f} (0=diversified, 1=monopoly)")
            print(f"    Shocked supplier share: {supplier_metrics['shocked_supplier_share']:.1%}")

            if indicators_df is not None and target_node in indicators_df.index:
                indicators = indicators_df.loc[target_node]
                econ_parts = []
                if 'log_gdp_per_capita' in indicators.index:
                    econ_parts.append(f"log_GDP/cap={indicators['log_gdp_per_capita']:.2f}")
                if 'gdp_growth' in indicators.index:
                    econ_parts.append(f"GDP_growth={indicators['gdp_growth']:.1f}%")
                if 'inflation' in indicators.index:
                    econ_parts.append(f"inflation={indicators['inflation']:.1f}%")
                if 'unemployment_rate' in indicators.index:
                    econ_parts.append(f"unemployment={indicators['unemployment_rate']:.1f}%")
                # if econ_parts:
                    # print(f"    Economic indicators ({baseline_year}): {', '.join(econ_parts)}")
            else:
                print("    Economic indicators: Not available")

            series = get_import_shock_series(
                api=self.api,
                is_domestic=is_domestic,
                target_country=target_country,
                target_node=target_node,
                shocked_country=shocked_country,
                shocked_sector=shocked_sector,
                shocked_node=shocked_node,
                shock_year=shock_year,
                shock_month=shock_month,
                baseline_year=baseline_year,
                shock_graph=shock_graph,
                baseline_graph=baseline_graph,
            )
            if series is None:
                continue
            country_imports_shock, country_imports_baseline, shock_keys_12, imports_all = series

            # Detect recovery month using already-fetched series (skip shock month itself)
            recovery_month = None
            for i, date_key in enumerate(shock_keys_12):
                if i == 0:
                    continue
                _month = date_key.split("-")[1]
                baseline_key = f"{baseline_year}-{_month}"
                if date_key in country_imports_shock and baseline_key in country_imports_baseline:
                    shock_val = country_imports_shock[date_key]
                    base_val = country_imports_baseline[baseline_key]
                    if base_val > 0:
                        yoy = (shock_val - base_val) / base_val
                        if yoy >= 0.0:
                            recovery_month = date_key
                            print(f"      ✓ Recovery detected at {date_key} (imports {yoy:+.1%})")
                            break
            
            if not country_imports_shock or not country_imports_baseline:
                print(f"    ⚠️  No import data available, skipping")
                continue
            
            if not recovery_month:
                print(f"    ⚠️  No recovery within 12 months")
            
            # Use a fixed 12-month window unless recovery happened earlier
            months_to_query = 12
            if recovery_month in shock_keys_12:
                months_to_query = shock_keys_12.index(recovery_month) + 1
            
            # print(f"    [WEIGHT] Calculating industry weight for {shocked_node} to {target_node}...")
            if is_domestic:
                # Domestic shock series is already the direct ICIO edge (industry-specific),
                # so do NOT re-weight it (avoid double allocation).
                icio_weight = 1.0
                print("      Domestic flow: using icio_weight=100.0%")
            else:
                icio_weight = self.icio.calculate_industry_weight(target_node, shocked_node, shock_graph)
            
            # Query PROPAGATION VALUE (target industry's total exports)
            # One query: 2 years pre-shock (same month) through 12 months after the shock month

            print(
                f"    [PROPAGATION] Querying {target_node} total exports "
                f"(from {shock_year-2}-{shock_month:02d} for 3 years)..."
            )
            industry_exports_all = self.api.get_trade_data(
                reporter=target_country,
                partner="WLD",
                sector_code=target_sector,
                flow_code="X",
                start_year=shock_year - 2,
                start_month=shock_month,
                duration_months=36,
                verbose=True,
            )
            if industry_exports_all:
                export_years = sorted({int(k.split("-")[0]) for k in industry_exports_all.keys() if "-" in k})
                print(f"      Export data years: {export_years}")

            shock_keys = month_keys(shock_year, shock_month, months_to_query)
            industry_exports_shock, industry_exports_baseline = slice_shock_and_baseline(
                industry_exports_all, shock_keys, baseline_year
            )
            
            if not industry_exports_shock or not industry_exports_baseline:
                print(f"    ⚠️  No export data available for propagation measurement, skipping")
                continue

            # Seasonal decomposition expectations (fit on 3 years pre-shock)
            # Also request expected values for t-12 months so we can compute expected YoY growth deviations.
            lag_keys = [f"{int(k.split('-')[0]) - 1}-{k.split('-')[1]}" for k in shock_keys]
            forecast_keys_ext = sorted(set(shock_keys + lag_keys))

            import_expected_map, import_resid_std = expected_from_pre_shock(
                series_all=imports_all,
                pre_keys=pre_keys,
                forecast_keys=forecast_keys_ext,
            )
            export_expected_map, export_resid_std = expected_from_pre_shock(
                series_all=industry_exports_all,
                pre_keys=pre_keys,
                forecast_keys=forecast_keys_ext,
            )
            
            # Extract training samples (month-by-month until shock recovery)
            # Iterate through all months where we have data
            all_months = sorted(set(country_imports_shock.keys()) & set(industry_exports_shock.keys()))
            
            for date_key in all_months:
                if recovery_month and date_key > recovery_month:
                    break  # Stop after shock recovers (A→B returns to baseline)
                
                year, month = date_key.split('-')
                # Use same month from baseline year
                baseline_key = f"{baseline_year}-{month}"
                lag_key = f"{int(year) - 1}-{month}"
                
                # Check if we have all required data for this month
                if baseline_key not in industry_exports_baseline:
                    continue
                if date_key not in country_imports_shock or baseline_key not in country_imports_baseline:
                    continue
                
                # SHOCK VALUE (shared logic for domestic/foreign; only the data source differs)
                import_shock = country_imports_shock[date_key]
                import_baseline = country_imports_baseline[baseline_key]
                if import_baseline <= 0:
                    continue
                import_yoy_change = (import_shock - import_baseline) / import_baseline
                weighted_shock_value = (import_shock - import_baseline) * icio_weight

                import_expected = import_expected_map.get(date_key)
                import_expected_lag = import_expected_map.get(lag_key)
                shock_dev_abs = (import_shock - import_expected) if import_expected is not None else None
                shock_dev_pct = (shock_dev_abs / import_expected) if (import_expected and import_expected > 0 and shock_dev_abs is not None) else None
                shock_dev_z = (shock_dev_abs / import_resid_std) if (shock_dev_abs is not None and import_resid_std and import_resid_std > 0) else None
                shock_dev_abs_weighted = (shock_dev_abs * icio_weight) if shock_dev_abs is not None else None

                # YoY growth deviation (actual YoY vs expected YoY using t-12)
                import_lag = imports_all.get(lag_key)
                shock_yoy_actual = (import_shock - import_lag) / import_lag if (import_lag is not None and import_lag > 0) else None
                shock_yoy_expected = (import_expected - import_expected_lag) / import_expected_lag if (import_expected is not None and import_expected_lag is not None and import_expected_lag > 0) else None
                shock_yoy_dev = (shock_yoy_actual - shock_yoy_expected) if (shock_yoy_actual is not None and shock_yoy_expected is not None) else None

                # YoY log-diff deviation
                shock_logyoy_actual = (math.log1p(import_shock) - math.log1p(import_lag)) if (import_lag is not None and import_lag >= 0) else None
                shock_logyoy_expected = (math.log1p(import_expected) - math.log1p(import_expected_lag)) if (import_expected is not None and import_expected_lag is not None and import_expected >= 0 and import_expected_lag >= 0) else None
                shock_logyoy_dev = (shock_logyoy_actual - shock_logyoy_expected) if (shock_logyoy_actual is not None and shock_logyoy_expected is not None) else None
                
                # Calculate PROPAGATION VALUE (export change) - THIS IS THE TARGET
                export_shock = industry_exports_shock[date_key]
                export_baseline = industry_exports_baseline[baseline_key]
                if export_baseline > 0:
                    propagation_value = (export_shock - export_baseline) / export_baseline
                else:
                    continue

                export_expected = export_expected_map.get(date_key)
                export_expected_lag = export_expected_map.get(lag_key)
                prop_dev_abs = (export_shock - export_expected) if export_expected is not None else None
                prop_dev_pct = (prop_dev_abs / export_expected) if (export_expected and export_expected > 0 and prop_dev_abs is not None) else None
                prop_dev_z = (prop_dev_abs / export_resid_std) if (prop_dev_abs is not None and export_resid_std and export_resid_std > 0) else None

                export_lag = industry_exports_all.get(lag_key)
                prop_yoy_actual = (export_shock - export_lag) / export_lag if (export_lag is not None and export_lag > 0) else None
                prop_yoy_expected = (export_expected - export_expected_lag) / export_expected_lag if (export_expected is not None and export_expected_lag is not None and export_expected_lag > 0) else None
                prop_yoy_dev = (prop_yoy_actual - prop_yoy_expected) if (prop_yoy_actual is not None and prop_yoy_expected is not None) else None

                prop_logyoy_actual = (math.log1p(export_shock) - math.log1p(export_lag)) if (export_lag is not None and export_lag >= 0) else None
                prop_logyoy_expected = (math.log1p(export_expected) - math.log1p(export_expected_lag)) if (export_expected is not None and export_expected_lag is not None and export_expected >= 0 and export_expected_lag >= 0) else None
                prop_logyoy_dev = (prop_logyoy_actual - prop_logyoy_expected) if (prop_logyoy_actual is not None and prop_logyoy_expected is not None) else None
                
                # Create training sample
                sample = {
                    'shock_event': shock['name'],
                    'shock_node': shocked_node,
                    'shock_date': shock_date,
                    'target_node': target_node,
                    'target_country': target_country,
                    'observation_date': date_key,
                    'months_after_shock': (int(year) - shock_year) * 12 + (int(month) - shock_month),
                    
                    # SHOCK VALUE (input)
                    'shock_yoy_change': import_yoy_change,
                    'shock_value': weighted_shock_value,
                    'shock_expected': import_expected,
                    'shock_dev_abs': shock_dev_abs,
                    'shock_dev_pct': shock_dev_pct,
                    'shock_dev_z': shock_dev_z,
                    'shock_dev_abs_weighted': shock_dev_abs_weighted,
                    'shock_resid_std': import_resid_std,
                    'shock_yoy_actual': shock_yoy_actual,
                    'shock_yoy_expected': shock_yoy_expected,
                    'shock_yoy_dev': shock_yoy_dev,
                    'shock_logyoy_actual': shock_logyoy_actual,
                    'shock_logyoy_expected': shock_logyoy_expected,
                    'shock_logyoy_dev': shock_logyoy_dev,
                    
                    # PROPAGATION VALUE (output/target)
                    'export_baseline': export_baseline,  # Industry B baseline exports
                    'export_observed': export_shock,  # Industry B observed exports
                    'yoy_change': propagation_value,  
                    'prop_expected': export_expected,
                    'prop_dev_abs': prop_dev_abs,
                    'prop_dev_pct': prop_dev_pct,
                    'prop_dev_z': prop_dev_z,
                    'prop_resid_std': export_resid_std,
                    'prop_yoy_actual': prop_yoy_actual,
                    'prop_yoy_expected': prop_yoy_expected,
                    'prop_yoy_dev': prop_yoy_dev,
                    'prop_logyoy_actual': prop_logyoy_actual,
                    'prop_logyoy_expected': prop_logyoy_expected,
                    'prop_logyoy_dev': prop_logyoy_dev,
                    
                    'icio_edge_value': partner['edge_value'],
                    'is_domestic': is_domestic,
                }
                
                # Add supplier metrics (already computed above)
                sample['supplier_hhi'] = supplier_metrics['supplier_hhi']
                sample['shocked_supplier_share'] = supplier_metrics['shocked_supplier_share']
                
                # Add economic indicators
                if indicators_df is not None and target_node in indicators_df.index:
                    indicators = indicators_df.loc[target_node]
                    for col in ['log_gdp_per_capita', 'gdp_growth', 'inflation', 'unemployment_rate']:
                        if col in indicators.index:
                            sample[f'target_{col}'] = indicators[col]
                
                training_samples.append(sample)
                print(f"      {date_key}: Import YoY={import_yoy_change:+.1%} → Export YoY={propagation_value:+.1%}")
                shock_dev_pct_str = "NA" if shock_dev_pct is None else f"{shock_dev_pct:+.1%}"
                shock_dev_z_str = "NA" if shock_dev_z is None else f"{shock_dev_z:+.2f}"
                prop_dev_pct_str = "NA" if prop_dev_pct is None else f"{prop_dev_pct:+.1%}"
                prop_dev_z_str = "NA" if prop_dev_z is None else f"{prop_dev_z:+.2f}"
                print(f"     Deviations: shock_pct={shock_dev_pct_str}, shock_z={shock_dev_z_str} | prop_pct={prop_dev_pct_str}, prop_z={prop_dev_z_str}")

                shock_yoy_dev_str = "NA" if shock_yoy_dev is None else f"{shock_yoy_dev:+.1%}"
                prop_yoy_dev_str = "NA" if prop_yoy_dev is None else f"{prop_yoy_dev:+.1%}"
                shock_logyoy_dev_str = "NA" if shock_logyoy_dev is None else f"{shock_logyoy_dev:+.3f}"
                prop_logyoy_dev_str = "NA" if prop_logyoy_dev is None else f"{prop_logyoy_dev:+.3f}"
                print(f"     Growth devs: shock_yoy_dev={shock_yoy_dev_str}, shock_logyoy_dev={shock_logyoy_dev_str} | prop_yoy_dev={prop_yoy_dev_str}, prop_logyoy_dev={prop_logyoy_dev_str}")
        
        print(f"\n✓ Extracted {len(training_samples)} training samples from this shock")
        return training_samples
    
    def build_dataset(self, shock_events: List[Dict], output_path: Path) -> pd.DataFrame:
        """
        Build complete training dataset from all shock events
        
        Saves incrementally after each shock event to prevent data loss.
        
        Args:
            shock_events: List of shock event definitions
            output_path: Path to save CSV incrementally
        
        Returns:
            DataFrame with all training samples
        """
        all_samples: List[Dict] = []

        # Clear output file if it exists (start fresh)
        if output_path.exists():
            output_path.unlink()
            print(f"Cleared existing output file: {output_path}")

        for i, shock in enumerate(shock_events, 1):
            try:
                print(f"\n[{i}/{len(shock_events)}] Processing shock event...")
                samples = self.process_shock_event(shock)

                if samples:
                    samples_df = pd.DataFrame(samples)
                    mode = "w" if i == 1 else "a"
                    header = i == 1
                    samples_df.to_csv(output_path, mode=mode, header=header, index=False)
                    print(f"✓ Saved {len(samples)} samples to {output_path} (mode={mode})")
                    all_samples.extend(samples)
                else:
                    print("⚠️  No samples generated for this shock event")

            except Exception as e:
                print(f"\n❌ Error processing shock '{shock['name']}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_samples:
            print("\n⚠️  No samples generated from any shock event")
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        
        print(f"\n{'='*80}")
        print("DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"Total training samples: {len(df)}")
        print(f"Shock events processed: {df['shock_event'].nunique()}")
        print(f"Unique target countries: {df['target_country'].nunique()}")
        print(f"Date range: {df['observation_date'].min()} to {df['observation_date'].max()}")
        print(f"Mean YoY change: {df['yoy_change'].mean():.1%}")
        print(f"Median YoY change: {df['yoy_change'].median():.1%}")
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Build shock propagation training dataset from UN Comtrade API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--shocks', required=True,
                       help='Path to shock events JSON file')
    parser.add_argument('--icio-dir', default='embeddings',
                       help='Directory containing ICIO graph files')
    parser.add_argument('--output', default='shocks/training_data.csv',
                       help='Output path for training dataset')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top FOREIGN downstream partners to track')
    
    args = parser.parse_args()
    
    # Load shock events
    print(f"Loading shock events from {args.shocks}...")
    with open(args.shocks, 'r') as f:
        shock_events = json.load(f)
    
    print(f"Loaded {len(shock_events)} shock events")
    
    # Prepare output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build dataset (saves incrementally)
    builder = ShockDatasetBuilder(
        icio_dir=args.icio_dir, 
        top_k=args.top_k,
    )
    dataset = builder.build_dataset(shock_events, output_path)
    
    if not dataset.empty:
        print(f"\n✓ Final dataset saved to: {output_path}")
        print(f"  Rows: {len(dataset)}")
        print(f"  Columns: {list(dataset.columns)}")
    else:
        print(f"\n⚠️  No data saved (empty dataset)")


if __name__ == "__main__":
    main()

