#!/usr/bin/env python3
"""
orchestrate_combo_train_and_live_eval.py

Implements an (optionally exhaustive) search over training event subsets:
1) For each event combination (size >= min_combo_size), train a model using
   train_propagation_model.py on only those events; keep only if selected MAE <= threshold.
2) For each kept model, run a live_partner_impact-equivalent scenario (same parameters as
   simulation_results_v4.csv), with early stopping: if hop-0 relative MAE on DEU_C19+ITA_C19
   is worse than the best so far, skip hop-1 propagation.
3) Keep the model with best relative MAE on the evaluation set excluding China + downstream.

Notes / realism:
- Exhaustive over all combos for N=17 events means 130,918 trainings; that is likely infeasible.
- This script includes guardrails: max_combos and max_combo_size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_EXCLUDE_NODES = ["CHN_C19", "VNM_C19", "PHL_G", "HKG_G", "SGP_H50"]


@dataclass(frozen=True)
class ScenarioSpec:
    shock_node: str
    shock_year: int
    shock_month: int
    shock_yoy_change: float
    months_after_shock: int
    candidates: int
    hops: int


def _sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _combo_id(events: Sequence[str]) -> str:
    key = "|".join(sorted(map(str, events)))
    return _sha1_short(key, 12)


def _rel_mae(y_pred: pd.Series, y_true: pd.Series, *, eps: float) -> Optional[float]:
    mask = y_pred.notna() & y_true.notna()
    if not bool(mask.any()):
        return None
    yp = y_pred.loc[mask].astype(float)
    yt = y_true.loc[mask].astype(float)
    denom = yt.abs().clip(lower=float(eps))
    return float((yp - yt).abs().div(denom).mean())


def _load_events(training_csv: Path) -> List[str]:
    df = pd.read_csv(training_csv, usecols=["shock_event"])
    return sorted(set(df["shock_event"].astype(str)))


def _total_combos(n: int, min_k: int, max_k: int) -> int:
    return int(sum(math.comb(n, k) for k in range(min_k, max_k + 1)))


def _iter_combos(events: List[str], *, min_k: int, max_k: int) -> Iterable[Tuple[str, ...]]:
    import itertools

    for k in range(min_k, max_k + 1):
        for combo in itertools.combinations(events, k):
            yield tuple(combo)


def _write_subset_csv(training_csv: Path, events: Sequence[str], out_csv: Path) -> int:
    df = pd.read_csv(training_csv)
    keep = set(map(str, events))
    sub = df[df["shock_event"].astype(str).isin(keep)].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return int(len(sub))


def _parse_selected_mae(model_dir: Path, *, select_by: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns (selected_model_name, selected_mae_mean) from train_propagation_model outputs.
    """
    schema_path = model_dir / "feature_schema.json"
    metrics_path = model_dir / "metrics.json"
    if not schema_path.exists() or not metrics_path.exists():
        return None, None
    schema = json.loads(schema_path.read_text())
    selected = schema.get("selected_model")
    metrics = json.loads(metrics_path.read_text())
    if selected is None:
        return None, None
    block = metrics.get("models", {}).get(str(selected), {})
    if select_by not in {"groupkfold", "leave_one_event_out"}:
        raise ValueError("--select-by must be groupkfold or leave_one_event_out")
    mae = (
        block.get(select_by, {})
        .get("summary", {})
        .get("mae", {})
        .get("mean")
    )
    try:
        return str(selected), float(mae)
    except Exception:
        return str(selected), None


def _train_subset(
    *,
    subset_csv: Path,
    outdir: Path,
    select_by: str,
    cv_folds: int,
    event_balanced_weighting: bool,
    seed: int,
) -> int:
    """
    Executes train_propagation_model.py as a subprocess.
    Returns process exit code.
    """
    train_py = SCRIPT_DIR / "train_propagation_model.py"
    cmd = [
        sys.executable,
        str(train_py),
        "--data",
        str(subset_csv),
        "--outdir",
        str(outdir),
        "--seed",
        str(int(seed)),
        "--cv-folds",
        str(int(cv_folds)),
        "--select-by",
        str(select_by),
    ]
    if event_balanced_weighting:
        cmd.append("--event-balanced-weighting")
    # Keep logs tidy; caller can inspect artifacts if needed.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (outdir / "train_log.txt").write_text(proc.stdout)
    return int(proc.returncode)


def _infer_scenario_from_simulation_csv(sim_csv: Path, *, default_candidates: int) -> ScenarioSpec:
    df = pd.read_csv(sim_csv)
    if "hop" in df.columns:
        df0 = df[df["hop"].astype(int) == 0].copy()
    else:
        df0 = df.copy()
    # assume first row corresponds to CLI shock definition
    r0 = df0.iloc[0]
    shock_node = str(r0["shock_node"])
    shock_year = int(r0["shock_year"])
    shock_month = int(r0["shock_month"])
    shock_yoy_change = float(r0["shock_yoy_actual"])
    months_after_shock = int(r0["months_after_shock"])
    hops = int(df["hop"].max()) + 1 if "hop" in df.columns else 1
    return ScenarioSpec(
        shock_node=shock_node,
        shock_year=shock_year,
        shock_month=shock_month,
        shock_yoy_change=shock_yoy_change,
        months_after_shock=months_after_shock,
        candidates=int(default_candidates),
        hops=int(hops),
    )


def _filter_eval_rows(
    df: pd.DataFrame,
    *,
    exclude_target_country: Sequence[str],
    exclude_target_nodes: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    if "target_country" in out.columns and exclude_target_country:
        out = out[~out["target_country"].astype(str).isin(set(map(str, exclude_target_country)))]
    if "target_node" in out.columns and exclude_target_nodes:
        out = out[~out["target_node"].astype(str).isin(set(map(str, exclude_target_nodes)))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train models on event combos and evaluate via live scenario.")
    ap.add_argument("--training-csv", default=str(SCRIPT_DIR / "training_data_v4.csv"))
    ap.add_argument("--min-combo-size", type=int, default=5)
    ap.add_argument("--max-combo-size", type=int, default=5, help="Guardrail; set to 17 for exhaustive.")
    ap.add_argument("--max-combos", type=int, default=500, help="Guardrail to prevent accidental exhaustive runs.")
    ap.add_argument("--force-exhaustive", action="store_true", help="Allow running beyond --max-combos.")
    ap.add_argument(
        "--search-mode",
        choices=["exhaustive", "beam"],
        default="exhaustive",
        help="Search strategy: exhaustive combinations vs greedy beam search.",
    )
    ap.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="For --search-mode=beam: number of partial sets to keep each round.",
    )
    ap.add_argument(
        "--beam-direction",
        choices=["forward", "reverse"],
        default="forward",
        help="For --search-mode=beam: forward adds events from small->large; reverse removes events from large->small.",
    )

    ap.add_argument("--select-by", choices=["groupkfold", "leave_one_event_out"], default="leave_one_event_out")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--event-balanced-weighting", action="store_true")
    ap.add_argument("--mae-threshold", type=float, default=0.3)

    ap.add_argument("--simulation-csv", default=str(SCRIPT_DIR / "simulation_results_v4.csv"))
    ap.add_argument("--candidates", type=int, default=5, help="Scenario candidates; simulation_results_v4.csv does not store this.")
    ap.add_argument("--epsilon", type=float, default=1e-9)
    ap.add_argument("--no-network", action="store_true", help="Disable Comtrade/WorldBank calls (evaluation will likely be NA).")

    ap.add_argument("--stop-on-deu-ita", action="store_true")
    ap.add_argument("--eval-focus-nodes", default="DEU_C19,ITA_C19")
    ap.add_argument("--exclude-target-country", default="CHN")
    ap.add_argument("--exclude-target-nodes", default=",".join(DEFAULT_EXCLUDE_NODES))

    ap.add_argument("--work-dir", default=str(SCRIPT_DIR / "combo_search_runs"))
    ap.add_argument("--results-csv", default=str(SCRIPT_DIR / "combo_search_results.csv"))
    ap.add_argument(
        "--resume-from",
        default="",
        help="Path to an existing results CSV to resume from (especially useful for --search-mode=beam). "
        "If set, this file will be used as the output file (appended) as well.",
    )
    ap.add_argument("--keep-all-models", action="store_true", help="Keep all passing models; otherwise keep only the best and delete the rest.")
    args = ap.parse_args()

    training_csv = Path(args.training_csv)
    sim_csv = Path(args.simulation_csv)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    events = _load_events(training_csv)
    n = len(events)
    min_k = int(args.min_combo_size)
    max_k = min(int(args.max_combo_size), n)
    if min_k < 1 or max_k < min_k:
        raise SystemExit("Invalid combo size bounds.")
    if str(args.search_mode) == "exhaustive":
        total = _total_combos(n, min_k, max_k)
        if (not bool(args.force_exhaustive)) and total > int(args.max_combos):
            raise SystemExit(
                f"Planned combos={total} exceeds --max-combos={args.max_combos}. "
                f"Increase --max-combo-size/--max-combos or pass --force-exhaustive."
            )

    scenario = _infer_scenario_from_simulation_csv(sim_csv, default_candidates=int(args.candidates))

    # Import inside main (keeps module import fast if user just wants counts)
    from live_partner_impact import ScenarioInputs, run_scenario

    best_rel_mae_focus = float("inf")
    best_rel_mae_overall = float("inf")
    best_model_dir: Optional[Path] = None
    best_row: Optional[dict] = None

    resume_from = Path(str(args.resume_from)) if str(args.resume_from).strip() else None

    results_path = Path(args.results_csv)
    if resume_from is not None:
        results_path = resume_from
        print(f"[resume] Using results file: {results_path}")
    else:
        # Avoid corrupting an existing CSV with different columns when switching modes.
        # Only auto-suffix if user is using the DEFAULT file.
        default_results = (SCRIPT_DIR / "combo_search_results.csv").resolve()
        if results_path.exists() and str(args.search_mode) == "beam" and results_path.resolve() == default_results:
            # Use a separate results file for beam runs unless user explicitly points elsewhere.
            alt = results_path.parent / f"{results_path.stem}_beam{results_path.suffix}"
            results_path = alt
            print(f"[beam] Writing results to: {results_path}")
    existing_done: set[str] = set()
    if results_path.exists():
        try:
            old = pd.read_csv(results_path, usecols=["combo_id"])
            existing_done = set(old["combo_id"].astype(str))
        except Exception:
            existing_done = set()

    rows: List[Dict[str, object]] = []

    focus_nodes = [s.strip() for s in str(args.eval_focus_nodes).split(",") if s.strip()]
    exclude_countries = [s.strip() for s in str(args.exclude_target_country).split(",") if s.strip()]
    exclude_nodes = [s.strip() for s in str(args.exclude_target_nodes).split(",") if s.strip()]

    # If resuming, restore best comparators from prior results (important for early stopping).
    resume_df: Optional[pd.DataFrame] = None
    if results_path.exists():
        try:
            resume_df = pd.read_csv(results_path)
        except Exception:
            resume_df = None
    if resume_df is not None and len(resume_df):
        if "search_mode" in resume_df.columns:
            resume_df = resume_df[resume_df["search_mode"].astype(str) == str(args.search_mode)].copy()
        if "beam_direction" in resume_df.columns:
            resume_df = resume_df[resume_df["beam_direction"].astype(str) == str(args.beam_direction)].copy()
        if "rel_mae_focus" in resume_df.columns and "passed_train" in resume_df.columns and "early_stopped" in resume_df.columns:
            m = (
                resume_df["passed_train"].astype(bool)
                & (~resume_df["early_stopped"].astype(bool))
                & pd.to_numeric(resume_df["rel_mae_focus"], errors="coerce").notna()
            )
            if bool(m.any()):
                best_rel_mae_focus = float(pd.to_numeric(resume_df.loc[m, "rel_mae_focus"], errors="coerce").min())
        if "rel_mae_overall" in resume_df.columns and "passed_train" in resume_df.columns and "early_stopped" in resume_df.columns:
            m = (
                resume_df["passed_train"].astype(bool)
                & (~resume_df["early_stopped"].astype(bool))
                & pd.to_numeric(resume_df["rel_mae_overall"], errors="coerce").notna()
            )
            if bool(m.any()):
                best_rel_mae_overall = float(pd.to_numeric(resume_df.loc[m, "rel_mae_overall"], errors="coerce").min())
                tmp = resume_df.loc[m].copy()
                tmp["__rel_mae_overall_num"] = pd.to_numeric(tmp["rel_mae_overall"], errors="coerce")
                tmp = tmp.sort_values(["__rel_mae_overall_num", "combo_id"])
                best_row = tmp.iloc[0].drop(columns=["__rel_mae_overall_num"]).to_dict()
                try:
                    best_model_dir = Path(str(best_row.get("model_dir", ""))) if str(best_row.get("model_dir", "")).strip() else None
                except Exception:
                    best_model_dir = None

    def _evaluate_combo(combo: Tuple[str, ...], *, phase: str) -> Dict[str, object]:
        cid = _combo_id(combo)
        combo_dir = work_dir / f"combo_{cid}"
        model_dir = combo_dir / "model"
        subset_csv = combo_dir / "subset.csv"

        combo_dir.mkdir(parents=True, exist_ok=True)
        n_rows = _write_subset_csv(training_csv, combo, subset_csv)

        rc = _train_subset(
            subset_csv=subset_csv,
            outdir=model_dir,
            select_by=str(args.select_by),
            cv_folds=int(args.cv_folds),
            event_balanced_weighting=bool(args.event_balanced_weighting),
            seed=int(args.seed),
        )
        selected_model, selected_mae = _parse_selected_mae(model_dir, select_by=str(args.select_by))
        passed_train = (rc == 0) and (selected_mae is not None) and (float(selected_mae) <= float(args.mae_threshold))

        early_stopped = False
        rel_mae_focus: Optional[float] = None
        rel_mae_overall: Optional[float] = None
        n_eval_rows: int = 0

        if passed_train and phase == "full":
            inputs0 = ScenarioInputs(
                shock_node=str(scenario.shock_node),
                shock_year=int(scenario.shock_year),
                shock_month=int(scenario.shock_month),
                shock_yoy_actual=float(scenario.shock_yoy_change),
                months_after_shock=int(scenario.months_after_shock),
                history_months=24,
                candidates=int(scenario.candidates),
            )
            df0 = run_scenario(
                model_dir=model_dir,
                training_csv=subset_csv,
                embeddings_dir=Path(SCRIPT_DIR.parent / "embeddings"),
                inputs=inputs0,
                use_network=not bool(args.no_network),
                hop=0,
                shock_is_dev=False,
            )
            df_focus = df0[df0["target_node"].astype(str).isin(set(focus_nodes))].copy()
            rel_mae_focus = _rel_mae(df_focus.get("y_pred"), df_focus.get("y_observed"), eps=float(args.epsilon))  # type: ignore[arg-type]
            if rel_mae_focus is None:
                rel_mae_focus = float("inf")

            if bool(args.stop_on_deu_ita) and float(rel_mae_focus) > float(best_rel_mae_focus):
                early_stopped = True
            else:
                dfs = [df0]
                if int(scenario.hops) >= 2:
                    active_nodes: List[str] = []
                    for _, r in df0.iterrows():
                        dev = r.get("pred_prop_yoy_dev")
                        if dev is None or pd.isna(dev) or float(dev) >= 0:
                            continue
                        active_nodes.append(str(r["target_node"]))
                    seen = set()
                    active_nodes = [x for x in active_nodes if not (x in seen or seen.add(x))]

                    for shock_node in active_nodes:
                        hop_inputs = ScenarioInputs(
                            shock_node=str(shock_node),
                            shock_year=int(scenario.shock_year),
                            shock_month=int(scenario.shock_month),
                            shock_yoy_actual=float(
                                df0.loc[df0["target_node"].astype(str) == str(shock_node), "pred_prop_yoy_dev"].iloc[0]
                            ),
                            months_after_shock=int(scenario.months_after_shock),
                            history_months=24,
                            candidates=int(scenario.candidates),
                        )
                        df1 = run_scenario(
                            model_dir=model_dir,
                            training_csv=subset_csv,
                            embeddings_dir=Path(SCRIPT_DIR.parent / "embeddings"),
                            inputs=hop_inputs,
                            use_network=not bool(args.no_network),
                            hop=1,
                            shock_is_dev=True,
                        )
                        dfs.append(df1)

                df_all = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else df0
                df_eval = _filter_eval_rows(
                    df_all,
                    exclude_target_country=exclude_countries,
                    exclude_target_nodes=exclude_nodes,
                )
                rel_mae_overall = _rel_mae(df_eval.get("y_pred"), df_eval.get("y_observed"), eps=float(args.epsilon))  # type: ignore[arg-type]
                n_eval_rows = int((df_eval.get("y_pred").notna() & df_eval.get("y_observed").notna()).sum())

        if (not passed_train) or (passed_train and (phase != "keep")):
            # Keep only what caller decides; delete failing subsets to save disk.
            if not passed_train:
                try:
                    shutil.rmtree(combo_dir)
                except Exception:
                    pass

        return {
            "combo_id": cid,
            "events": "|".join(combo),
            "n_events": int(len(combo)),
            "n_rows": int(n_rows),
            "train_exit_code": int(rc),
            "selected_model": selected_model or "",
            "selected_mae": float(selected_mae) if selected_mae is not None else float("nan"),
            "passed_train": bool(passed_train),
            "early_stopped": bool(early_stopped),
            "rel_mae_focus": float(rel_mae_focus) if rel_mae_focus is not None else float("nan"),
            "rel_mae_overall": float(rel_mae_overall) if rel_mae_overall is not None else float("nan"),
            "n_eval_rows": int(n_eval_rows),
            "model_dir": str(model_dir),
            "search_mode": str(args.search_mode),
            "beam_direction": str(args.beam_direction) if str(args.search_mode) == "beam" else "",
            "round_k": int(len(combo)),
        }

    if str(args.search_mode) == "exhaustive":
        for combo in _iter_combos(events, min_k=min_k, max_k=max_k):
            cid = _combo_id(combo)
            if cid in existing_done:
                continue
            row = _evaluate_combo(combo, phase="full")

            # update best focus comparator + best overall if applicable
            if bool(row["passed_train"]) and (not bool(row["early_stopped"])) and (not math.isnan(float(row["rel_mae_focus"]))):
                best_rel_mae_focus = min(best_rel_mae_focus, float(row["rel_mae_focus"]))
            if bool(row["passed_train"]) and (not bool(row["early_stopped"])) and (not math.isnan(float(row["rel_mae_overall"]))):
                if float(row["rel_mae_overall"]) < float(best_rel_mae_overall):
                    best_rel_mae_overall = float(row["rel_mae_overall"])
                    best_model_dir = Path(row["model_dir"])
                    best_row = dict(row)

            rows.append(row)
            if len(rows) >= 10:
                df_out = pd.DataFrame(rows)
                write_header = not results_path.exists()
                df_out.to_csv(results_path, mode="a", index=False, header=write_header)
                rows = []
    else:
        # Beam search (greedy forward selection keeping top-N partial sets each round).
        beam_width = int(args.beam_width)
        if beam_width < 1:
            raise SystemExit("--beam-width must be >= 1")

        def _seed_beam_by_rowcount() -> List[Tuple[str, ...]]:
            # Round-1 (single event) cannot be trained with train_propagation_model.py because grouped CV
            # requires >=2 unique shock events. So we seed the beam using a cheap proxy: row count per event.
            df_ev = pd.read_csv(training_csv, usecols=["shock_event"])
            event_counts = df_ev["shock_event"].astype(str).value_counts().to_dict()
            singletons = [(e,) for e in events]
            singletons.sort(key=lambda t: int(event_counts.get(t[0], 0)), reverse=True)
            return singletons[:beam_width]

        def _seed_full_beam_by_rowcount(*, k: int) -> List[Tuple[str, ...]]:
            """
            Heuristic seed for reverse beam: create beam_width distinct k-sized sets from the top events by row count.
            """
            if k < 1:
                return []
            df_ev = pd.read_csv(training_csv, usecols=["shock_event"])
            counts = df_ev["shock_event"].astype(str).value_counts().to_dict()
            ranked = sorted(events, key=lambda e: int(counts.get(e, 0)), reverse=True)
            # Create sliding windows so we get different sets without randomness.
            pool = ranked[: max(k + beam_width, k)]
            out: List[Tuple[str, ...]] = []
            for i in range(beam_width):
                start = i % max(1, len(pool))
                # wrap-around window
                window = [pool[(start + j) % len(pool)] for j in range(k)]
                out.append(tuple(sorted(set(window))))
            # ensure correct sizes; if set() dedup shrank, top-up with ranked events
            fixed: List[Tuple[str, ...]] = []
            for t in out:
                s = list(t)
                if len(s) < k:
                    for e in ranked:
                        if e in s:
                            continue
                        s.append(e)
                        if len(s) == k:
                            break
                fixed.append(tuple(sorted(s[:k])))
            return sorted(set(fixed))[:beam_width]

        def _reconstruct_beam_from_resume(df: pd.DataFrame) -> Tuple[List[Tuple[str, ...]], int]:
            """
            Reconstruct (beam, current_k) from a previous beam CSV.
            If reconstruction isn't possible, returns (seed_beam, 1).
            """
            if "round_k" not in df.columns or "events" not in df.columns:
                return _seed_beam_by_rowcount(), 1

            rk = pd.to_numeric(df["round_k"], errors="coerce")
            if not bool(rk.notna().any()):
                return _seed_beam_by_rowcount(), 1

            # Prefer the latest round that has at least one passing training row.
            current_k = int(rk.max())
            for kk in range(current_k, 1, -1):
                dfk = df[rk == float(kk)].copy()
                if "passed_train" in dfk.columns:
                    dfk = dfk[dfk["passed_train"].astype(bool)].copy()
                if len(dfk):
                    current_k = kk
                    break
            else:
                return _seed_beam_by_rowcount(), 1

            dfk = df[rk == float(current_k)].copy()
            dfk = dfk[dfk.get("passed_train", True).astype(bool)].copy()
            if len(dfk) == 0:
                return _seed_beam_by_rowcount(), 1

            if current_k < min_k:
                if "selected_mae" not in dfk.columns:
                    return _seed_beam_by_rowcount(), 1
                dfk["__score"] = pd.to_numeric(dfk["selected_mae"], errors="coerce")
            else:
                if "rel_mae_overall" not in dfk.columns:
                    return _seed_beam_by_rowcount(), 1
                dfk["__score"] = pd.to_numeric(dfk["rel_mae_overall"], errors="coerce")
                if "early_stopped" in dfk.columns:
                    dfk.loc[dfk["early_stopped"].astype(bool), "__score"] = float("inf")

            dfk = dfk[dfk["__score"].notna()].copy()
            if len(dfk) == 0:
                return _seed_beam_by_rowcount(), 1

            dfk = dfk.sort_values(["__score", "combo_id"])
            top = dfk.head(int(beam_width))
            beam = [tuple(str(s).split("|")) for s in top["events"].astype(str).tolist()]
            return beam, int(current_k)

        beam: List[Tuple[str, ...]]
        start_k = 2
        if resume_df is not None and len(resume_df):
            beam, current_k = _reconstruct_beam_from_resume(resume_df)
            if str(args.beam_direction) == "forward":
                start_k = max(2, int(current_k) + 1)
                if start_k > max_k:
                    print(f"[beam] Resume: already completed up to round_k={current_k} (max_k={max_k}). Nothing to do.")
                    beam = []
                else:
                    print(f"[beam] Resume: continuing from round_k={start_k} using reconstructed beam from round_k={current_k}.")
            else:
                # reverse: we will go downward, so next round is current_k - 1
                start_k = min(max_k, int(current_k) - 1)
                if start_k < min_k:
                    print(f"[beam] Resume: already completed down to round_k={current_k} (min_k={min_k}). Nothing to do.")
                    beam = []
                else:
                    print(f"[beam] Resume: continuing reverse from round_k={start_k} using reconstructed beam from round_k={current_k}.")
        else:
            if str(args.beam_direction) == "forward":
                beam = _seed_beam_by_rowcount()
                print(f"[beam] Seeded round-1 beam with top-{beam_width} events by row-count.")
            else:
                beam = _seed_full_beam_by_rowcount(k=max_k)
                start_k = max_k - 1
                print(f"[beam] Seeded reverse beam at k={max_k} with top-{beam_width} sets by row-count; starting at k={start_k}.")

        # Start expanding from size=2 (first size where training is feasible) or from resume point.
        if str(args.beam_direction) == "forward":
            k_iter = range(int(start_k), max_k + 1)
        else:
            # reverse direction: remove events until min_k
            k_iter = range(int(start_k), min_k - 1, -1)

        for k in k_iter:
            candidates: List[Tuple[str, ...]] = []
            for partial in beam:
                used = set(partial)
                if str(args.beam_direction) == "forward":
                    for e in events:
                        if e in used:
                            continue
                        cand = tuple(sorted(partial + (e,)))
                        candidates.append(cand)
                else:
                    # reverse: remove one element
                    for e in list(partial):
                        cand = tuple(sorted([x for x in partial if x != e]))
                        if len(cand) != k:
                            continue
                        candidates.append(cand)

            # Dedup
            candidates = sorted(set(candidates))
            if not candidates:
                break

            cand_rows: List[Dict[str, object]] = []
            for combo in candidates:
                cid = _combo_id(combo)
                if cid in existing_done:
                    continue

                phase = "train_only" if k < min_k else "full"
                row = _evaluate_combo(combo, phase="full" if phase == "full" else "keep")
                # In train_only rounds, we do NOT run live eval; keep rel_mae fields NaN.
                if phase == "train_only":
                    row["early_stopped"] = False
                    row["rel_mae_focus"] = float("nan")
                    row["rel_mae_overall"] = float("nan")
                    row["n_eval_rows"] = 0
                cand_rows.append(row)

                rows.append(row)
                if len(rows) >= 20:
                    df_out = pd.DataFrame(rows)
                    write_header = not results_path.exists()
                    df_out.to_csv(results_path, mode="a", index=False, header=write_header)
                    rows = []

            if not cand_rows:
                break

            # Filter to passing training rows
            passing = [r for r in cand_rows if bool(r["passed_train"])]
            if not passing:
                break

            # Score for beam selection:
            # - before min_k: use selected_mae (training quality proxy)
            # - at/after min_k: use rel_mae_overall (objective), with early-stopped rows penalized.
            def _score(r: Dict[str, object]) -> float:
                if k < min_k:
                    return float(r["selected_mae"])
                if bool(r["early_stopped"]):
                    return float("inf")
                v = float(r["rel_mae_overall"])
                return v if not math.isnan(v) else float("inf")

            passing.sort(key=_score)
            next_beam_rows = passing[:beam_width]
            beam = [tuple(str(r["events"]).split("|")) for r in next_beam_rows]

            # Update global bests when we have full eval
            if k >= min_k:
                for r in next_beam_rows:
                    if bool(r["early_stopped"]):
                        continue
                    if not math.isnan(float(r["rel_mae_focus"])):
                        best_rel_mae_focus = min(best_rel_mae_focus, float(r["rel_mae_focus"]))
                    if not math.isnan(float(r["rel_mae_overall"])) and float(r["rel_mae_overall"]) < float(best_rel_mae_overall):
                        best_rel_mae_overall = float(r["rel_mae_overall"])
                        best_model_dir = Path(r["model_dir"])
                        best_row = dict(r)

            # Cleanup: keep only combo dirs corresponding to beam + global best (unless keep-all-models)
            if not bool(args.keep_all_models):
                keep_ids = set(_combo_id(b) for b in beam)
                if best_row is not None:
                    keep_ids.add(str(best_row["combo_id"]))
                for r in passing:
                    cid = str(r["combo_id"])
                    if cid in keep_ids:
                        continue
                    d = work_dir / f"combo_{cid}"
                    try:
                        shutil.rmtree(d)
                    except Exception:
                        pass

    # flush remaining
    if rows:
        df_out = pd.DataFrame(rows)
        write_header = not results_path.exists()
        df_out.to_csv(results_path, mode="a", index=False, header=write_header)

    print(f"Done. Results: {results_path}")
    if best_row is not None:
        print(
            "Best model:\n"
            f"  events={best_row['events']}\n"
            f"  train_selected_mae={best_row['selected_mae']}\n"
            f"  rel_mae_focus(DEU_C19+ITA_C19)={best_row['rel_mae_focus']}\n"
            f"  rel_mae_overall(excl China+downstream)={best_row['rel_mae_overall']}\n"
            f"  model_dir={best_row['model_dir']}"
        )


if __name__ == "__main__":
    main()


