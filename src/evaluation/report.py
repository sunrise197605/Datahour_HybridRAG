"""
Report generation utilities.
Converts evaluation result rows into a pandas DataFrame for easy analysis
and viewing. Can be extended to export plots or formatted reports.
"""

from typing import Any, Dict

import pandas as pd


def build_summary_tables(results: Dict[str, Any]) -> pd.DataFrame:
    """Converts the per-question result rows into a DataFrame for analysis."""
    return pd.DataFrame(results.get("rows", []))
