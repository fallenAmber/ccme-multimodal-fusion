#!/usr/bin/env python3
"""
dedup_split_check.py
Standalone script (no notebook/torch dependency) that:
  1. Loads metadata.csv and pv_module_crops_v2/index_unlabeled.csv
     (UNLABELED scores - see preprocessing_verified.py's build_module_dataset)
  2. Groups near-duplicate frames (GPS/time proximity + perceptual image hash)
  3. Assigns groups to train/val/test using CROP VOLUME ONLY - no label or
     score information, since labels don't exist yet at this stage under
     the training-derived-threshold protocol (using labels here would be
     a form of test-distribution leakage)
  4. Verifies no group is split across partitions, and that none of the
     56 explicitly-excluded frames leaked through
  5. Derives the anomaly threshold from TRAINING scores only, freezes it,
     and applies it unchanged to validation/test (preprocessing_verified.
     assign_frozen_labels)
  6. Saves the result to pv_module_crops_v2/index_labeled.csv

Run it from the same folder as your notebook:
    python dedup_split_check.py

Then in the notebook, instead of recomputing the split, just load the
result of this script directly (see the printed instructions at the end).

Requires: pandas, numpy, scikit-learn, pillow, imagehash
    pip install imagehash
"""

from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import imagehash
except ImportError:
    sys.exit(
        "Missing dependency. Install it first:\n"
        "  pip install imagehash"
    )

DATA_ROOT = Path("/Users/mdtohidulislam/Documents/Research/new_dataset/datasets")
DR_META = DATA_ROOT / "double-row" / "metadata.csv"
DR_RGB_DIR = DATA_ROOT / "double-row" / "rgb images"
OUT_ROOT = Path("./pv_module_crops_v2")
IDX_PATH = OUT_ROOT / "index_unlabeled.csv"
OUT_PATH = OUT_ROOT / "index_labeled.csv"

SEED = 42
# Relaxed volume bounds (as fraction of total crops) - chosen to prioritize
# leakage-safety (1) and anomaly-rate balance (2) over exact 70/10/20 sizing
# (3), per the final protocol decision: anomaly-positive crops are highly
# concentrated in a small number of event groups, so rigid 70/10/20 sizing
# and near-uniform anomaly prevalence are not simultaneously achievable.
VOL_BOUNDS_FRAC = {
    "train": (0.60, 0.70),
    "val":   (0.10, 0.15),
    "test":  (0.20, 0.30),
}
N_RESTARTS = 60
ITERS_PER_RESTART = 3000
TIME_GAP_MAX = 0.5              # seconds; a real pause/reposition
GPS_RADIUS_DEG = 1.2e-4         # ~10-12m cumulative drift from block anchor
BOUNDARY_HASH_THRESHOLD = 7     # strict: near-identical only, checked at
                                 # coarse-block boundaries (not chained).
                                 # Calibrated against the real 122-frame
                                 # hover block (true boundary distance: 6).


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df = df.rename(columns={
        "rgb image name": "rgb_image_name",
        "thermal image name": "thermal_image_name",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["rgb_path"] = df["rgb_image_name"].apply(lambda x: DR_RGB_DIR / x)
    df["frame_num"] = df["rgb_image_name"].str.extract(r"(\d+)").astype(int)
    return df


def coarse_gps_time_groups(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: cheap coarse grouping using cumulative distance from a
    moving anchor (resets on a real time gap or enough total drift)."""
    meta_df = meta_df.sort_values("frame_num").reset_index(drop=True)
    block_id = 0
    anchor_lon = meta_df.loc[0, "longitude"]
    anchor_lat = meta_df.loc[0, "latitude"]
    last_t = meta_df.loc[0, "timestamp"]
    ids = [0]
    for i in range(1, len(meta_df)):
        row = meta_df.loc[i]
        dt = (row["timestamp"] - last_t).total_seconds()
        dist = max(abs(row["longitude"] - anchor_lon), abs(row["latitude"] - anchor_lat))
        if dt > TIME_GAP_MAX or dist > GPS_RADIUS_DEG:
            block_id += 1
            anchor_lon, anchor_lat = row["longitude"], row["latitude"]
        ids.append(block_id)
        last_t = row["timestamp"]
    meta_df["coarse_block"] = ids
    return meta_df


def refine_with_perceptual_hash(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: at each COARSE-BLOCK BOUNDARY only (where Stage 1 decided to
    start a new block due to GPS drift/time gap), check whether the two
    frames straddling that boundary are near-identical. If so, merge the
    blocks - this catches genuine hovers that a fixed GPS radius clips.

    IMPORTANT: this does NOT chain-merge through ordinary consecutive-frame
    similarity. At normal frame rates, adjacent frames of continuous flight
    are almost always similar regardless of whether the drone is hovering
    or cruising - chaining on that basis wrongly merges long stretches of
    ordinary flight over visually repetitive content (e.g. rows of panels)
    into one artificial block. Restricting the check to boundary pairs only,
    with a strict threshold, targets real revisits/hovers specifically.
    """
    frame_nums = meta_df["frame_num"].tolist()
    coarse = meta_df["coarse_block"].tolist()

    # Only need to hash frames sitting at a coarse-block boundary (first/last
    # frame of each coarse block) - much cheaper than hashing everything,
    # and avoids the chaining failure mode entirely.
    boundary_idx = set()
    for i in range(1, len(meta_df)):
        if coarse[i] != coarse[i - 1]:
            boundary_idx.add(i - 1)  # last frame of outgoing block
            boundary_idx.add(i)      # first frame of incoming block

    print(f"Hashing {len(boundary_idx)} boundary frames "
          f"(out of {len(meta_df)} total - boundary-only, not all frames)...")
    hashes = {}
    n_failed = 0
    for i in sorted(boundary_idx):
        row = meta_df.loc[i]
        try:
            hashes[i] = imagehash.phash(Image.open(row["rgb_path"]))
        except Exception:
            n_failed += 1
            hashes[i] = None
    if n_failed:
        print(f"  [warn] {n_failed} boundary images could not be opened/hashed")

    group_ids = [0]
    cur_group = 0
    for i in range(1, len(meta_df)):
        if coarse[i] == coarse[i - 1]:
            group_ids.append(cur_group)
            continue
        # coarse boundary -- check if it's actually a real content break
        h_prev, h_cur = hashes.get(i - 1), hashes.get(i)
        near_identical = (
            h_prev is not None and h_cur is not None
            and (h_prev - h_cur) <= BOUNDARY_HASH_THRESHOLD
        )
        if near_identical:
            group_ids.append(cur_group)  # merge across the boundary
        else:
            cur_group += 1
            group_ids.append(cur_group)
    meta_df["group_id"] = group_ids
    return meta_df


def assign_volume_only_split(index_df: pd.DataFrame, vol_bounds_frac: dict, seed: int = 42) -> dict:
    """
    Assigns whole groups to train/val/test based ONLY on crop volume -
    no reference to labels or scores. Use this instead of
    balance_split_by_rate() when following the training-derived-threshold
    protocol, where labels do not exist yet at split time (using labels
    to balance the split before the threshold is established would itself
    be a form of test-distribution leakage). Some anomaly-prevalence
    difference across splits is expected and accepted under this
    protocol, not optimized away - report it, don't hide it.

    Returns {group_id: split_name}.
    """
    random.seed(seed)
    splits = list(vol_bounds_frac.keys())
    g = index_df.groupby("group_id")["image_index"].count().reset_index(name="total_crops")
    total = g["total_crops"].sum()
    vol_bounds = {k: (lo * total, hi * total) for k, (lo, hi) in vol_bounds_frac.items()}
    gids = g.sort_values("total_crops", ascending=False)["group_id"].tolist()
    size = dict(zip(g["group_id"], g["total_crops"]))

    cur = {k: 0 for k in splits}
    assign = {}
    mid = {k: (vol_bounds[k][0] + vol_bounds[k][1]) / 2 for k in splits}
    for gid in gids:  # largest groups first (greedy bin-packing, LPT heuristic)
        cands = [k for k in splits if cur[k] + size[gid] <= vol_bounds[k][1]] or splits
        k = min(cands, key=lambda k: cur[k] - mid[k])
        assign[gid] = k
        cur[k] += size[gid]
    return assign


def balance_split_by_rate(
    index_df: pd.DataFrame,
    vol_bounds_frac: dict,
    n_restarts: int = 60,
    iters_per_restart: int = 3000,
    seed: int = 42,
) -> dict:
    """
    LEGACY - kept only for historical comparison against the earlier
    (global-quantile-labeled) protocol. DO NOT use with the
    training-derived-threshold protocol: this function requires a
    'label' column, which does not exist at split time under that
    protocol, and using labels to balance the split before the
    threshold is established would itself be a form of
    test-distribution leakage. Use assign_volume_only_split() instead.
    """
    raise RuntimeError(
        "balance_split_by_rate() is legacy and incompatible with the "
        "training-derived-threshold protocol (it requires labels, which "
        "don't exist yet at split time). Use assign_volume_only_split() "
        "instead. If you specifically want the old global-quantile-label "
        "protocol for comparison, call _balance_split_by_rate_legacy_impl() "
        "directly."
    )


def _balance_split_by_rate_legacy_impl(
    index_df: pd.DataFrame,
    vol_bounds_frac: dict,
    n_restarts: int = 60,
    iters_per_restart: int = 3000,
    seed: int = 42,
) -> dict:
    """
    Assigns whole groups to train/val/test to keep anomaly PREVALENCE close
    to the dataset-wide rate in every split, subject to relaxed volume
    bounds (fraction of total crops per split). Random-restart local search
    (swap-based) - not provably optimal, but empirically closes the gap to
    near-zero rate deviation on this dataset's group structure.

    Returns {group_id: split_name}.
    """
    random.seed(seed)
    splits = list(vol_bounds_frac.keys())
    g = (
        index_df.groupby("group_id")
        .agg(total_crops=("label", "size"), positive_crops=("label", "sum"))
        .reset_index()
    )
    total_crops = g["total_crops"].sum()
    overall_rate = g["positive_crops"].sum() / total_crops
    vol_bounds = {
        k: (lo * total_crops, hi * total_crops) for k, (lo, hi) in vol_bounds_frac.items()
    }
    gids = g["group_id"].tolist()
    pos = dict(zip(g["group_id"], g["positive_crops"]))
    size = dict(zip(g["group_id"], g["total_crops"]))

    def get_stats(assign):
        cur_pos = {k: 0 for k in splits}
        cur_size = {k: 0 for k in splits}
        for gid, sp in assign.items():
            cur_pos[sp] += pos[gid]
            cur_size[sp] += size[gid]
        return cur_pos, cur_size

    def feasible(cur_size):
        return all(vol_bounds[k][0] <= cur_size[k] <= vol_bounds[k][1] for k in splits)

    def rate_dev(cur_pos, cur_size):
        return sum(
            abs((cur_pos[k] / cur_size[k] if cur_size[k] > 0 else 0) - overall_rate)
            for k in splits
        )

    def random_feasible_init():
        order = gids[:]
        random.shuffle(order)
        assign, cur_size = {}, {k: 0 for k in splits}
        mid = {k: (vol_bounds[k][0] + vol_bounds[k][1]) / 2 for k in splits}
        for gid in order:
            candidates = [k for k in splits if cur_size[k] + size[gid] <= vol_bounds[k][1]]
            if not candidates:
                candidates = splits
            k_choice = min(candidates, key=lambda k: cur_size[k] - mid[k])
            assign[gid] = k_choice
            cur_size[k_choice] += size[gid]
        return assign

    def local_search(assign, max_iters):
        cur_pos, cur_size = get_stats(assign)
        best_dev = rate_dev(cur_pos, cur_size) if feasible(cur_size) else float("inf")
        ids = list(assign.keys())
        for _ in range(max_iters):
            i, j = random.sample(ids, 2)
            if assign[i] == assign[j]:
                continue
            assign[i], assign[j] = assign[j], assign[i]
            cp, cs = get_stats(assign)
            dev = rate_dev(cp, cs) if feasible(cs) else float("inf")
            if dev < best_dev:
                best_dev = dev
            else:
                assign[i], assign[j] = assign[j], assign[i]
        return assign, best_dev

    best_overall, best_overall_dev = None, None
    for _ in range(n_restarts):
        assign = random_feasible_init()
        assign, dev = local_search(assign, iters_per_restart)
        if best_overall_dev is None or dev < best_overall_dev:
            best_overall_dev, best_overall = dev, dict(assign)

    return best_overall


def main():
    print(f"Loading metadata from: {DR_META}")
    if not DR_META.exists():
        sys.exit(f"metadata.csv not found at {DR_META} -- edit DATA_ROOT at the top of this script.")
    meta = load_metadata(DR_META)
    print(f"  {len(meta)} frames loaded")

    required = ["frame_num", "timestamp", "longitude", "latitude", "rgb_path"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if meta["frame_num"].duplicated().any():
        dupes = meta.loc[meta["frame_num"].duplicated(), "frame_num"].tolist()
        raise ValueError(f"Duplicate frame numbers found: {dupes[:20]}")
    if meta[["timestamp", "longitude", "latitude"]].isna().any().any():
        bad_rows = meta[meta[["timestamp", "longitude", "latitude"]].isna().any(axis=1)]
        raise ValueError(
            f"{len(bad_rows)} metadata rows have missing/invalid timestamp or GPS values."
        )
    print("  Metadata validation passed.")

    print(f"\nLoading unlabeled crop index from: {IDX_PATH}")
    if not IDX_PATH.exists():
        sys.exit(f"index_unlabeled.csv not found at {IDX_PATH} -- run "
                  f"preprocessing_verified.build_module_dataset() first, "
                  f"or edit OUT_ROOT at the top of this script.")
    index_df = pd.read_csv(IDX_PATH)
    print(f"  {len(index_df)} crops loaded")

    print("\n=== Stage 1: GPS/time coarse grouping ===")
    meta = coarse_gps_time_groups(meta)
    print(f"  {meta['coarse_block'].nunique()} coarse groups")

    print("\n=== Stage 2: perceptual-hash refinement ===")
    meta = refine_with_perceptual_hash(meta)
    n_groups = meta["group_id"].nunique()
    sizes = meta.groupby("group_id").size()
    print(f"  {n_groups} final groups "
          f"(size min={sizes.min()} median={sizes.median():.0f} "
          f"max={sizes.max()} mean={sizes.mean():.1f})")

    frame_to_group = dict(zip(meta["frame_num"], meta["group_id"]))
    index_df["group_id"] = index_df["image_index"].map(frame_to_group)
    n_unmapped = index_df["group_id"].isna().sum()
    if n_unmapped:
        print(f"  [warn] {n_unmapped} crops have no group mapping -- dropping them")
        index_df = index_df.dropna(subset=["group_id"])
    index_df["group_id"] = index_df["group_id"].astype(int)

    print("\n=== Building volume-only group-level split ===")
    print(f"  Volume bounds: { {k: tuple(round(v*100,1) for v in vb) for k,vb in VOL_BOUNDS_FRAC.items()} } "
          f"(% of total crops)")
    print("  (No label/score information used -- labels don't exist yet "
          "under the training-derived-threshold protocol.)")
    best_assign = assign_volume_only_split(index_df, VOL_BOUNDS_FRAC, seed=SEED)

    index_df["split"] = index_df["group_id"].map(best_assign)
    if index_df["split"].isna().any():
        n_bad = index_df["split"].isna().sum()
        raise RuntimeError(f"{n_bad} crops were not assigned to a split.")

    crop_frac = index_df["split"].value_counts(normalize=True)
    for split_name, (lo, hi) in VOL_BOUNDS_FRAC.items():
        frac = crop_frac.get(split_name, 0.0)
        if not (lo <= frac <= hi):
            raise RuntimeError(
                f"{split_name} crop fraction {frac:.4f} is outside "
                f"the allowed range [{lo:.2f}, {hi:.2f}]."
            )
    print("  PASSED: all split crop fractions are within the requested bounds.")

    print("\n=== Verifying leakage-safety ===")
    checks = index_df.groupby("group_id")["split"].nunique()
    bad = checks[checks > 1]
    if len(bad):
        raise RuntimeError(f"{len(bad)} groups span multiple splits: {bad.index.tolist()}")
    print("  PASSED: every group is fully contained within a single split.")

    print("\n=== Unlabeled split summary ===")
    summary = index_df.groupby("split").agg(
        n_crops=("image_index", "size"),
        n_groups=("group_id", "nunique"),
        n_frames=("image_index", "nunique"),
    )
    summary["crop_frac"] = summary["n_crops"] / len(index_df)
    print(summary)

    # Confirm none of the 56 explicitly-excluded frames leaked through
    print("\n=== Explicit exclusion verification ===")
    excluded_present = index_df[
        index_df["image_index"].between(0, 32)
        | index_df["image_index"].between(2518, 2540)
    ]
    if len(excluded_present):
        raise RuntimeError(
            f"Excluded-frame leakage detected: "
            f"{sorted(excluded_present['image_index'].unique())}"
        )
    print("  PASSED: no crops from the 56 excluded frames are present.")

    # 122-frame duplicate-block check (unrelated to the exclusion above --
    # this is the near-duplicate hover-block leakage check from the
    # original leakage audit)
    print("\n=== 122-frame duplicate-block check (should be single-split) ===")
    sub = index_df[index_df["image_index"].between(1689, 1810)]
    splits_here = sub["split"].unique()
    if len(splits_here) > 1:
        raise RuntimeError(f"122-frame block spans multiple splits: {list(splits_here)}")
    print("  PASSED: 122-frame block is contained within one split.")

    # Labels are assigned HERE, after the split, using training scores only
    print("\n=== Assigning frozen, training-derived labels ===")
    from preprocessing_verified import assign_frozen_labels
    index_df = assign_frozen_labels(index_df, OUT_ROOT, top_frac=0.12)
    # assign_frozen_labels already saved index_labeled.csv to OUT_ROOT;
    # OUT_PATH matches that same location for consistency with older calls.

    print(f"\nSaved: {OUT_PATH}")
    print(
        "\nTo use this in your notebook instead of recomputing the split, "
        "load it directly:\n"
        "  index_df = pd.read_csv('pv_module_crops_v2/index_labeled.csv')\n"
        "  train_df = index_df[index_df['split']=='train']\n"
        "  val_df   = index_df[index_df['split']=='val']\n"
        "  test_df  = index_df[index_df['split']=='test']\n"
        "  # then build PairedModuleDataset(train_df, ...), etc. as before"
    )


if __name__ == "__main__":
    main()
