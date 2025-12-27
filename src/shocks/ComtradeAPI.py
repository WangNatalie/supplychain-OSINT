import os
import time
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import comtradeapicall

class ComtradeAPI:
    """UN Comtrade API wrapper using comtradeapicall package"""
    
    def __init__(self, rate_limit_delay=1.0):
        """
        Args:
            rate_limit_delay: Seconds to wait between API calls
        
        Note:
            Subscription key is read from COMTRADE_API_KEY environment variable
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        self.subscription_key = os.getenv('COMTRADE_API_KEY')
        
        # Load ISO3 to M49 mapping from countries.csv
        self.iso3_to_m49 = self._load_country_mappings()
        
        # Load ICIO sector to HS code mappings
        self.sector_to_hs = self._load_sector_to_hs_mappings()

    def _rate_limit(self):
        """Basic client-side rate limiting between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call_time = time.time()

    def get_hs_codes_for_sector(self, sector_code: str) -> Optional[str]:
        """Return comma-separated HS4 codes for an ICIO sector code, or None if missing."""
        hs_codes = self.sector_to_hs.get(sector_code)
        if not hs_codes:
            return None
        return ",".join(hs_codes)
    
    def _load_country_mappings(self) -> Dict[str, str]:
        """Load ISO3 to M49 mapping from countries.csv"""
        mapping = {}
        countries_path = Path(__file__).parent.parent / 'ICIO' / 'countries.csv'
        
        try:
            df = pd.read_csv(countries_path)
            for _, row in df.iterrows():
                iso3 = row['Code']
                m49 = str(row['M49'])
                mapping[iso3] = m49
            
            print(f"✓ Loaded {len(mapping)} country codes from {countries_path.name}")
        except Exception as e:
            print(f"⚠️  Failed to load country mappings: {e}")
            print(f"   Using empty mapping (API calls may fail)")
        
        return mapping
    
    def _load_sector_to_hs_mappings(self) -> Dict[str, List[str]]:
        """
        Load ICIO sector to HS code mappings via ISIC Rev.4
        
        Mapping chain: ICIO sector → ISIC Rev.4 → HS4 codes
        Example: C29 → 2910 → [8702, 8703, 8704, ...]
        
        Returns:
            Dict mapping ICIO sector codes to lists of HS4 codes
        """
        icio_base = Path(__file__).parent.parent / 'ICIO'
        industries_path = icio_base / 'industries.csv'
        h4_to_isic_path = Path(__file__).parent.parent / 'ports' / 'H4_to_ISIC.csv'
        
        try:
            # Step 1: Load ICIO sector → ISIC Rev.4 mapping
            industries_df = pd.read_csv(industries_path)
            sector_to_isic = {}
            for _, row in industries_df.iterrows():
                sector_code = row['Code']
                isic_code = str(row['ISIC Rev.4']).strip()
                
                # Handle ranges like "69 to 75"
                if 'to' in isic_code:
                    start, end = isic_code.split('to')
                    isic_codes = [str(i) for i in range(int(start.strip()), int(end.strip()) + 1)]
                else:
                    isic_codes = [isic_code]
                
                if sector_code not in sector_to_isic:
                    sector_to_isic[sector_code] = []
                sector_to_isic[sector_code].extend(isic_codes)
            
            # Step 2: Load ISIC Rev.4 → HS4 mapping
            h4_df = pd.read_csv(h4_to_isic_path)
            isic_to_hs = {}
            for _, row in h4_df.iterrows():
                hs4 = str(row['HS4']).strip()
                isic_full = str(row['ISIC Rev. 4']).strip()
                
                # Extract first 2-3 digits for matching
                # ISIC codes like "2910" → match to "29" or "2910"
                if isic_full not in isic_to_hs:
                    isic_to_hs[isic_full] = set()
                isic_to_hs[isic_full].add(hs4)
            
            # Step 3: Combine mappings: ICIO sector → HS codes
            sector_to_hs = {}
            for sector, isic_list in sector_to_isic.items():
                hs_codes = set()
                for isic in isic_list:
                    # Try exact match first, then prefix match
                    for isic_key in isic_to_hs:
                        if isic_key.startswith(isic) or isic_key == isic:
                            hs_codes.update(isic_to_hs[isic_key])
                
                if hs_codes:
                    sector_to_hs[sector] = sorted(hs_codes)
            
            print(f"✓ Loaded sector→HS mappings for {len(sector_to_hs)} ICIO sectors")
            return sector_to_hs
            
        except Exception as e:
            print(f"⚠️  Failed to load sector→HS mappings: {e}")
            print(f"   Will query TOTAL trade (less precise)")
            return {}
    
    def _iso3_to_m49(self, iso3_code: str) -> str:
        """Convert ISO3 country code to M49 numeric code"""
        if iso3_code == 'WLD':
            return '0'  # World
        return self.iso3_to_m49.get(iso3_code, iso3_code)
    
    def get_trade_data(
        self,
        reporter: str,
        partner: str,
        *,
        sector_code: Optional[str] = None,
        commodity_code: Optional[str] = None,
        flow_code: str = "M",
        start_year: int,
        start_month: int,
        duration_months: int,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Query UN Comtrade for a monthly time series using comtradeapicall.
        
        Args:
            reporter: ISO3 country code (e.g., 'JPN') - will convert to M49
            partner: ISO3 country code or 'WLD' for world - will convert to M49
            sector_code: ICIO sector code (e.g., 'C29'); mapped to HS4 codes internally
            commodity_code: HS code string (or comma-separated list). If provided, overrides sector_code.
            flow_code: 'X' for exports, 'M' for imports
            start_year: Start year (inclusive)
            start_month: Start month (1-12, inclusive)
            duration_months: Number of months to query
            verbose: Whether to print HS code usage
        
        Returns:
            Dict mapping 'YYYY-MM' -> trade value in USD (missing months omitted)
        """
        # Determine commodity codes (sector_code -> HS4 list) unless overridden by commodity_code
        cmd_code: Optional[str] = commodity_code
        if cmd_code is None and sector_code:
            cmd_code = self.get_hs_codes_for_sector(sector_code)

        if verbose and sector_code and cmd_code:
            print(f"      Using HS codes for {sector_code}: {cmd_code[:50]}...")

        # Build periods list: YYYYMM,YYYYMM,...
        periods: List[str] = []
        y = start_year
        m = start_month
        for _ in range(duration_months):
            periods.append(f"{y}{m:02d}")
            m += 1
            if m > 12:
                m = 1
                y += 1

        # Convert ISO3 to M49 codes
        reporter_m49 = self._iso3_to_m49(reporter)
        partner_m49 = self._iso3_to_m49(partner)

        # Comtrade often rejects long comma-separated period lists; chunk to 12 months (1 year) per request.
        chunk_size = 12
        out: Dict[str, float] = {}

        for i in range(0, len(periods), chunk_size):
            chunk = periods[i:i + chunk_size]
            self._rate_limit()

            try:
                df = comtradeapicall.getFinalData(
                    subscription_key=self.subscription_key,
                    typeCode='C',  # Commodities
                    freqCode='M',  # Monthly
                    clCode='HS',   # Harmonized System
                    period=",".join(chunk),
                    reporterCode=reporter_m49,
                    cmdCode=cmd_code if cmd_code else 'TOTAL',
                    flowCode=flow_code,
                    partnerCode=partner_m49,
                    partner2Code=None,
                    customsCode=None,
                    motCode=None,
                    maxRecords=5000,
                    format_output='JSON',
                    aggregateBy=None,
                    breakdownMode='classic',
                    countOnly=None,
                    includeDesc=True
                )

                if df is None or df.empty:
                    continue

                value_col = None
                for col in ['primaryValue', 'TradeValue', 'fobvalue', 'cifvalue']:
                    if col in df.columns:
                        value_col = col
                        break
                if not value_col:
                    print(f"  Warning: No value column found for {reporter}<-{partner} (periods={len(chunk)})")
                    continue

                if "period" not in df.columns:
                    # Can't map to months; skip
                    continue

                for period_str, g in df.groupby("period"):
                    p = str(period_str)
                    if len(p) < 6:
                        continue
                    yy = int(p[:4])
                    mm = int(p[4:6])
                    date_key = f"{yy}-{mm:02d}"
                    v = float(g[value_col].sum())
                    if v > 0:
                        out[date_key] = v

            except Exception as e:
                # Continue other chunks
                print(f"  API error for {reporter}<-{partner} (period chunk starting {chunk[0]}): {e}")
                continue

        return out