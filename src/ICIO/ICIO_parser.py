"""
ICIO Code Parser - Convert OECD ICIO codes to human-readable names

This module provides functions to convert country codes (e.g., 'USA') and 
industry/sector codes (e.g., 'A01') to plain English descriptions.

Usage:
    from ICIO.ICIO_parser import lookup_country, lookup_industry, format_node_name
    
    # Look up individual codes
    print(lookup_country('USA'))  # → 'United States'
    print(lookup_industry('A01')) # → 'Agriculture, hunting, forestry'
    
    # Format full node codes
    print(format_node_name('USA_A01'))  # → 'United States - Agriculture, hunting, forestry'
"""

import pandas as pd
from pathlib import Path

# Load the lookup CSV files
_module_dir = Path(__file__).parent
industry_codes = pd.read_csv(_module_dir / "industries.csv")
country_codes = pd.read_csv(_module_dir / "countries.csv")

# Pre-build lookup dictionaries for fast access 
_industry_dict = dict(zip(industry_codes["Code"], industry_codes["Industry"]))
_country_dict = dict(zip(country_codes["Code"], country_codes["Country"]))


# ============================================================================
# Basic Lookup Functions
# ============================================================================

def lookup_industry(code: str) -> str:
    """
    Lookup industry/sector code.
    
    Args:
        code: Industry code (e.g., 'A01', 'C10T12', 'G')
    
    Returns:
        Industry name or original code if not found
    
    Examples:
        >>> lookup_industry('A01')
        'Agriculture, hunting, forestry'
    """
    return _industry_dict.get(code, code)


def lookup_country(code: str) -> str:
    """
    Lookup country code.
    
    Args:
        code: Country code (e.g., 'USA', 'CHN', 'DEU')
    
    Returns:
        Country name or original code if not found
    
    Examples:
        >>> lookup_country('USA')
        'United States'
    """
    return _country_dict.get(code, code)


# ============================================================================
# Formatting Functions (for display in reports/output)
# ============================================================================

def format_node_name(node_code: str, include_code: bool = False) -> str:
    """
    Convert node code to plain English format.
    
    Args:
        node_code: Node ID like 'USA_A01' or 'CHN_C10T12'
        include_code: If True, append code in brackets
    
    Returns:
        Formatted string like 'United States - Agriculture, hunting, forestry'
    
    Examples:
        >>> format_node_name('USA_A01')
        'United States - Agriculture, hunting, forestry'
        >>> format_node_name('USA_A01', include_code=True)
        'United States - Agriculture, hunting, forestry [USA_A01]'
    """
    if '_' not in node_code:
        country_name = lookup_country(node_code)
        return f"{country_name} [{node_code}]" if include_code else country_name
    
    country_code, sector_code = node_code.split('_', 1)
    country_name = lookup_country(country_code)
    sector_name = lookup_industry(sector_code)
    
    result = f"{country_name} - {sector_name}"
    if include_code:
        result += f" [{node_code}]"
    
    return result


def format_sector_name(sector_code: str, include_code: bool = False) -> str:
    """
    Format sector code to plain English.
    
    Args:
        sector_code: Sector code (e.g., 'A01', 'C10T12')
        include_code: If True, append code in brackets
    
    Returns:
        Formatted sector name
    
    Examples:
        >>> format_sector_name('A01')
        'Agriculture, hunting, forestry'
    """
    sector_name = lookup_industry(sector_code)
    return f"{sector_name} [{sector_code}]" if include_code else sector_name


def format_country_name(country_code: str, include_code: bool = False) -> str:
    """
    Format country code to plain English.
    
    Args:
        country_code: Country code (e.g., 'USA', 'CHN')
        include_code: If True, append code in brackets
    
    Returns:
        Formatted country name
    
    Examples:
        >>> format_country_name('USA')
        'United States'
    """
    country_name = lookup_country(country_code)
    return f"{country_name} [{country_code}]" if include_code else country_name
