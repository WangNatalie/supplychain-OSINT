#!/usr/bin/env python3
"""
prefetch_comtrade_cache.py

Warm the Comtrade cache for the exact query shapes used by live_partner_impact.py,
so subsequent scenario/model evaluations reuse cached responses.

This prefetches:
- hop-0: shock_node -> top_k downstream partners
- hop-1 worst-case: for each hop-0 partner as a potential propagated shock node,
  prefetch its top_k downstream partners too.

It does NOT require a trained model (we assume worst-case hop-1 fanout).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ComtradeAPI import ComtradeAPI
from shock_helpers import ICIOHelper


def _parse_country(node: str) -> str:
    return str(node).split("_", 1)[0]


def _parse_sector(node: str) -> str:
    parts = str(node).split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _add_months(y: int, m: int, delta: int) -> Tuple[int, int]:
    total = (y * 12 + (m - 1)) + int(delta)
    ny = total // 12
    nm = (total % 12) + 1
    return int(ny), int(nm)


def _months_between_inclusive(start_y: int, start_m: int, end_y: int, end_m: int) -> int:
    s = start_y * 12 + (start_m - 1)
    e = end_y * 12 + (end_m - 1)
    if e < s:
        raise ValueError("end must be >= start")
    return (e - s) + 1


def _latest_graph_year(embeddings_dir: Path) -> int:
    yrs: List[int] = []
    for p in embeddings_dir.glob("graph_*_labeled.pt"):
        parts = p.stem.split("_")
        for token in parts:
            if token.isdigit() and len(token) == 4:
                yrs.append(int(token))
    if not yrs:
        raise FileNotFoundError(f"No ICIO graphs found in: {embeddings_dir}")
    return max(yrs)


def _prefetch_for_pair(
    api: ComtradeAPI,
    *,
    shock_node: str,
    target_node: str,
    target_country: str,
    target_sector: str,
    shock_year: int,
    shock_month: int,
    months_after_shock: int,
) -> None:
    shocked_country = _parse_country(shock_node)
    shocked_sector = _parse_sector(shock_node)

    obs_year, obs_month = _add_months(shock_year, shock_month, months_after_shock)
    baseline_year = obs_year - 1

    # 1) Import baseline (1 month)
    api.get_trade_data(
        reporter=target_country,
        partner=shocked_country,
        sector_code=shocked_sector,
        flow_code="M",
        start_year=int(baseline_year),
        start_month=int(obs_month),
        duration_months=1,
        verbose=False,
    )

    # 2) Import history (24 months pre-shock window)
    api.get_trade_data(
        reporter=target_country,
        partner=shocked_country,
        sector_code=shocked_sector,
        flow_code="M",
        start_year=int(shock_year - 2),
        start_month=int(shock_month),
        duration_months=24,
        verbose=False,
    )

    # 3) Export history up to obs month (from shock_year-2, shock_month)
    start_y = int(shock_year - 2)
    start_m = int(shock_month)
    duration_months = _months_between_inclusive(start_y, start_m, int(obs_year), int(obs_month))
    api.get_trade_data(
        reporter=target_country,
        partner="WLD",
        sector_code=target_sector,
        flow_code="X",
        start_year=int(start_y),
        start_month=int(start_m),
        duration_months=int(duration_months),
        verbose=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prefetch Comtrade cache for live_partner_impact scenario.")
    ap.add_argument("--embeddings-dir", default=str((Path(__file__).resolve().parents[1] / "embeddings").as_posix()))
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "comtrade_cache"))
    ap.add_argument("--shock-node", required=True)
    ap.add_argument("--shock-year", type=int, required=True)
    ap.add_argument("--shock-month", type=int, required=True)
    ap.add_argument("--shock-yoy-change", type=float, required=False, default=0.0, help="Unused; kept for parity with live_partner_impact CLI.")
    ap.add_argument("--months-after-shock", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2, help="Prefetch hop-0 and worst-case hop-1 if hops>=2.")
    ap.add_argument("--rate-limit-delay", type=float, default=1.0)
    ap.add_argument(
        "--out-pairs-csv",
        default="",
        help="Optional: write all (shock_node,target_node,...) pairs prefetched to this CSV path.",
    )
    ap.add_argument(
        "--print-shock-nodes",
        action="store_true",
        help="Print the expanded shock_nodes list used for prefetch (hop0 nodes included when hops>=2).",
    )
    ap.add_argument(
        "--print-pairs",
        type=int,
        default=0,
        help="Print the first N pairs that were prefetched (0 disables).",
    )
    args = ap.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    icio = ICIOHelper(embeddings_dir)
    icio_year = min(int(args.shock_year), _latest_graph_year(embeddings_dir))

    api = ComtradeAPI(
        rate_limit_delay=float(args.rate_limit_delay),
        enable_cache=True,
        cache_dir=Path(args.cache_dir),
        verbose_cache=True,
    )

    # hop-0 downstream partners for the provided shock
    hop0 = icio.get_downstream_partners(str(args.shock_node), icio_year, top_k=int(args.candidates))
    hop0_nodes = [str(p["target_node"]) for p in hop0]

    # Build list of (shock_node -> downstream partners)
    shock_nodes: List[str] = [str(args.shock_node)]
    if int(args.hops) >= 2:
        shock_nodes.extend(hop0_nodes)

    if bool(args.print_shock_nodes):
        print("Shock nodes to prefetch:")
        for s in shock_nodes:
            print(f"- {s}")

    total_pairs = 0
    pair_rows: List[Dict[str, object]] = []
    for s_node in shock_nodes:
        partners = icio.get_downstream_partners(str(s_node), icio_year, top_k=int(args.candidates))
        for p in partners:
            target_node = str(p["target_node"])
            target_country = str(p["target_country"])
            target_sector = str(p.get("target_sector") or _parse_sector(target_node))
            pair_rows.append(
                {
                    "shock_node": str(s_node),
                    "target_node": target_node,
                    "target_country": target_country,
                    "target_sector": target_sector,
                    "icio_year": int(icio_year),
                }
            )
            _prefetch_for_pair(
                api,
                shock_node=str(s_node),
                target_node=target_node,
                target_country=target_country,
                target_sector=target_sector,
                shock_year=int(args.shock_year),
                shock_month=int(args.shock_month),
                months_after_shock=int(args.months_after_shock),
            )
            total_pairs += 1

    if int(args.print_pairs) and pair_rows:
        n = min(int(args.print_pairs), len(pair_rows))
        print(f"\nFirst {n} prefetched pairs:")
        for r in pair_rows[:n]:
            print(f"- {r['shock_node']} -> {r['target_node']} ({r['target_country']}/{r['target_sector']})")

    if args.out_pairs_csv:
        out_path = Path(args.out_pairs_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pair_rows).to_csv(out_path, index=False)
        print(f"\nSaved pairs CSV: {out_path}")

    # Each pair triggers 3 Comtrade queries in this prefetch helper.
    est_requests = int(total_pairs) * 3
    print(
        f"Prefetch complete. shock_nodes={len(shock_nodes)} | total_pairs={total_pairs} | "
        f"est_comtrade_requests={est_requests} | cache_dir={Path(args.cache_dir)}"
    )


if __name__ == "__main__":
    main()


