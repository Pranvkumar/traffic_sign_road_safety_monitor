"""Leakage-safe, group-aware splitting.

Whole groups (drive sequence, physical-sign track, source image) are assigned to a
single split so near-identical neighbouring frames never cross train/val/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _assign_groups(sizes: pd.Series, fractions: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
    """Shuffle groups, then place largest first into the split furthest below target."""
    groups = np.array(sorted(sizes.index))
    rng.shuffle(groups)
    order = sorted(groups, key=lambda g: -int(sizes[g]))  # stable: ties keep shuffled order
    n_total = int(sizes.sum())
    names = list(fractions)
    targets = {name: fractions[name] * n_total for name in names}
    filled = {name: 0 for name in names}
    assignment: dict[str, str] = {}
    for g in order:
        name = min(names, key=lambda s: (filled[s] - targets[s]) / max(targets[s], 1e-9))
        assignment[g] = name
        filled[name] += int(sizes[g])
    return assignment


def group_split(df: pd.DataFrame, group_col: str, fractions: dict[str, float], seed: int = 42,
                only_split_value: str = "unassigned", stratify_col: str | None = None) -> pd.DataFrame:
    """Assign rows whose ``split == only_split_value`` to splits by group.

    Rows that already have a split (e.g. an official test set) are left untouched.
    Whole groups go to one split. With ``stratify_col`` the assignment is done
    separately inside each stratum (e.g. class), so every split sees every class;
    groups must then be nested inside strata.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not in manifest")
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    out = df.copy()
    pool = out[out["split"] == only_split_value]
    if pool.empty:
        raise ValueError(f"No rows with split == '{only_split_value}' to assign")
    if (pool[group_col].astype(str) == "").any():
        raise ValueError(f"Empty group IDs in column '{group_col}'; cannot split safely")

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    if stratify_col:
        if (pool.groupby(group_col)[stratify_col].nunique() > 1).any():
            raise ValueError(f"Groups in '{group_col}' span several '{stratify_col}' values")
        for _, part in pool.groupby(stratify_col):
            assignment.update(_assign_groups(part.groupby(group_col).size(), fractions, rng))
    else:
        assignment = _assign_groups(pool.groupby(group_col).size(), fractions, rng)

    mask = out["split"] == only_split_value
    out.loc[mask, "split"] = out.loc[mask, group_col].map(assignment)
    return out


def check_group_disjoint(df: pd.DataFrame, group_col: str) -> dict[str, list[str]]:
    """Return groups that appear in more than one split (empty dict == leakage-free)."""
    per_group = df.groupby(group_col)["split"].nunique()
    leaked = per_group[per_group > 1].index.tolist()
    return {"leaked_groups": leaked}


def split_summary(df: pd.DataFrame, group_col: str, label_col: str | None = None) -> dict:
    summary = {}
    for split, part in df.groupby("split"):
        item = {"samples": int(len(part)), "groups": int(part[group_col].nunique())}
        if label_col and label_col in part.columns:
            item["classes"] = int(part[label_col].nunique())
        summary[str(split)] = item
    return summary
