"""
preprocessing_verified.py

Corrected module segmentation + pseudo-labeling. Fixes three bugs found by
review in the previous inline notebook version:

  1. STATISTICAL-LEVEL MISMATCH: the baseline was computed by pooling all
     accepted panels' pixels together, but each panel was scored using its
     own padded-crop P95 - mixing a pixel-level baseline with a per-panel
     numerator. FIXED: each accepted panel's T95 is computed once, from its
     own mask. The baseline (median, MAD) is computed across THOSE panel-
     level T95 values - true panel-vs-other-panels comparison.

  2. PADDED BBOX CONTAMINATING THE SCORE: the padded rectangular crop (used
     for saving training patches) was also used to compute the score,
     letting grass/gaps/neighboring structures leak into the statistic that
     determines the label. FIXED: T95 for scoring comes from the exact
     connected-component mask only. The padded crop is still saved as the
     model-input patch (it needs context), but never used for labeling.

  3. FALLBACK SILENTLY REINTRODUCED THE WHOLE-FRAME BASELINE: when too few
     panels were detected, the old code fell back to whole-frame stats,
     flagged but not fixed. FIXED: frames with fewer than MIN_VALID_PANELS
     accepted panels are EXCLUDED from pseudo-labeling entirely, and every
     exclusion is logged and reported, not silently patched over.

Also fixes: frame_num is extracted explicitly from the filename rather than
assumed to equal the DataFrame's positional index; FORCE_REBUILD properly
clears the output directory; every failed/excluded frame is logged, not
silently skipped.

NOTE ON THE SATURATION THRESHOLD: SAT_Z_THRESHOLD=-4.5 was evaluated in a
blinded manual audit of 3,530 candidate crops (1 marked uncertain, 3,529
scored). Of 3,412 manually confirmed panel candidates, 3,410 were accepted
and 2 rejected (panel recall = 99.9%). Of 117 confirmed non-panel
candidates, 80 were rejected and 37 accepted (non-panel rejection rate =
68.4%). False acceptance was concentrated in two known non-standard
acquisition intervals (17/61 in the checkerboard-calibration sequence,
19/54 in the vehicle/ground-inspection sequence) - see
EXCLUDED_FRAME_RANGES below, which excludes those intervals outright. The
threshold remains frozen at -4.5 (not retuned on this same audit set) and
is retained only as a secondary safeguard for the remaining frames, where
the audit's ordinary_flight sample showed non-panel objects to be rare.

"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

PREPROCESSING_VERSION = "v5_train_derived_threshold"
SAT_Z_THRESHOLD = -4.5
MIN_VALID_PANELS = 5  # frames with fewer accepted panels are excluded, not
                       # fallback-scored. Provisional - see note above.

# Frames excluded outright, based on a blinded audit of 3,530 candidates
# (3,529 scored; 1 marked uncertain).
# (see labeled_filter_evaluation.csv / score_saturation_filter output).
# The saturation filter alone retained 99.9% of genuine panels but only
# rejected 68.4% of non-panel candidates overall - and within these two
# specific known-contaminated windows, it missed 17/61 (checkerboard
# calibration sequence) and 19/54 (vehicle/ground-inspection sequence)
# non-panel candidates. Per-candidate filtering alone is not reliable
# enough for these two known windows; explicit exclusion is used instead.
# The saturation filter remains active for all other frames as a
# secondary safeguard (ordinary-flight frames showed very few non-panel
# candidates in the same audit, suggesting contamination is concentrated
# in these two windows rather than spread broadly across the dataset).
EXCLUDED_FRAME_RANGES = ((0, 32), (2518, 2540))


def is_excluded_frame(frame_num: int) -> bool:
    return any(lo <= frame_num <= hi for lo, hi in EXCLUDED_FRAME_RANGES)


def preprocess_rgb_hsv_otsu(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks = []
    for ci, ch in enumerate(cv2.split(hsv)):
        ch = cv2.GaussianBlur(ch, (3, 3), 0)
        _, mask = cv2.threshold(ch, 0, 179 if ci == 0 else 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(mask)
    return cv2.bitwise_and(cv2.bitwise_and(masks[0], masks[1]), masks[2])


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return ndi.binary_fill_holes(mask > 0).astype(np.uint8)


def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-6)


def bbox_from_label(labels, label_id):
    ys, xs = np.where(labels == label_id)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def extract_module_crops(
    rgb_crop_bgr, thermal8_crop, thermal_celsius_crop,
    min_area=500, pad=6,
    sat_z_threshold=SAT_Z_THRESHOLD, min_valid_panels=MIN_VALID_PANELS,
):
    """
    Returns (samples, mask01, labels, excluded_reason, diagnostics).
    excluded_reason is None if the frame was scored normally, or a string
    explaining why it was excluded (e.g. "too_few_panels: 2 < 5").
    diagnostics is always a dict (even on exclusion) so every exclusion is
    auditable: n_components_total, n_above_area_threshold, n_accepted,
    n_rejected, sat_median, sat_mad.
    """
    mask01 = postprocess_mask(preprocess_rgb_hsv_otsu(rgb_crop_bgr))
    n, labels = cv2.connectedComponents(mask01)
    hsv = cv2.cvtColor(rgb_crop_bgr, cv2.COLOR_BGR2HSV)
    H, W = thermal8_crop.shape

    candidates = []
    for lab in range(1, n):
        blob_mask = labels == lab
        area = int(blob_mask.sum())
        if area < min_area:
            continue
        bb = bbox_from_label(labels, lab)
        if bb is None:
            continue
        med_sat = float(np.median(hsv[:, :, 1][blob_mask]))
        candidates.append({"lab": lab, "area": area, "bbox": bb,
                            "med_sat": med_sat, "blob_mask": blob_mask})

    diag = {"n_components_total": n - 1, "n_above_area_threshold": len(candidates),
            "n_accepted": None, "n_rejected": None, "sat_median": None, "sat_mad": None}

    if len(candidates) == 0:
        diag.update(n_accepted=0, n_rejected=0)
        return [], mask01, labels, "no_candidates_detected", diag

    sats = np.array([c["med_sat"] for c in candidates])
    frame_med_sat, frame_mad_sat = np.median(sats), robust_mad(sats)
    diag.update(sat_median=float(frame_med_sat), sat_mad=float(frame_mad_sat))
    for c in candidates:
        c["sat_z"] = (c["med_sat"] - frame_med_sat) / frame_mad_sat

    accepted = [c for c in candidates if c["sat_z"] >= sat_z_threshold]
    diag.update(n_accepted=len(accepted), n_rejected=len(candidates) - len(accepted))

    if len(accepted) < min_valid_panels:
        return [], mask01, labels, f"too_few_panels: {len(accepted)} < {min_valid_panels}", diag

    # Panel-level T95 for EACH accepted panel, from its own mask only
    # (this is the statistic used for BOTH the baseline and each score -
    # true panel-vs-other-panels comparison, no pixel pooling).
    for c in accepted:
        panel_pixels = thermal_celsius_crop[c["blob_mask"]]
        c["t95"] = float(np.percentile(panel_pixels, 95))

    panel_t95 = np.array([c["t95"] for c in accepted])
    frame_panel_median = float(np.median(panel_t95))
    frame_panel_mad = robust_mad(panel_t95)

    samples = []
    for c in accepted:
        score = (c["t95"] - frame_panel_median) / frame_panel_mad
        x1, y1, x2, y2 = c["bbox"]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
        # Padded patch saved for model input ONLY - never used for scoring.
        # NOTE: no "label" is assigned here (the old k=6.0 threshold was
        # previously silently overwritten by build_module_dataset's
        # global top_frac quantile relabeling - keeping a fake label here
        # was misleading dead code. Labels are assigned ONCE, centrally,
        # after the dataset is fully built (see build_module_dataset).
        samples.append({
            "rgb_patch": rgb_crop_bgr[y1:y2, x1:x2],
            "th8_patch": thermal8_crop[y1:y2, x1:x2],
            "score": score,
            "bbox": (x1, y1, x2, y2), "area": c["area"],
            "sat_z": c["sat_z"], "t95": c["t95"],
            "n_accepted_panels": len(accepted),
        })
    return samples, mask01, labels, None, diag


def get_frame_num(rgb_path) -> int:
    """Extract the true frame number from the filename, rather than
    assuming it equals a DataFrame's positional index."""
    return int(Path(rgb_path).stem.split("_")[-1])


def save_patch(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def build_module_dataset(df, out_root: Path, load_rgb_fn, thermogram_cls,
                          make_aligned_pair_fn, crop_image_fn, ir_crop,
                          max_images=None, force_rebuild=False, tqdm_fn=None):
    """
    df must have 'rgb_path' and 'thermal_path' columns.
    load_rgb_fn, thermogram_cls, make_aligned_pair_fn, crop_image_fn, ir_crop
    are passed in from the notebook so this module has no hidden coupling
    to notebook-only globals.
    """
    import shutil

    out_root = Path(out_root)
    if force_rebuild and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rgb").mkdir(exist_ok=True)
    (out_root / "thermal").mkdir(exist_ok=True)

    records = []
    excluded = []
    failed = []
    df_iter = df.iloc[:max_images] if max_images else df
    iterator = tqdm_fn(df_iter.iterrows(), total=len(df_iter), desc="Building dataset") \
        if tqdm_fn else df_iter.iterrows()

    for _, row in iterator:
        frame_num = get_frame_num(row["rgb_path"])
        if is_excluded_frame(frame_num):
            excluded.append({"frame_num": frame_num,
                              "reason": "non_standard_acquisition_sequence",
                              "n_components_total": None, "n_above_area_threshold": None,
                              "n_accepted": None, "n_rejected": None,
                              "sat_median": None, "sat_mad": None})
            continue
        try:
            rgb = load_rgb_fn(Path(row["rgb_path"]))
            th = thermogram_cls(Path(row["thermal_path"]))
            th8 = th.render8()
            rgb_crop, th8_crop = make_aligned_pair_fn(rgb, th8)
            thC_crop = crop_image_fn(th.celsius, ir_crop)
        except Exception as e:
            failed.append({"frame_num": frame_num, "error": str(e)})
            continue

        samples, _, _, excl_reason, diag = extract_module_crops(rgb_crop, th8_crop, thC_crop)
        if excl_reason is not None:
            excluded.append({"frame_num": frame_num, "reason": excl_reason, **diag})
            continue

        for m, s in enumerate(samples):
            rp = out_root / "rgb" / f"img{frame_num:05d}_m{m:03d}.npy"
            tp = out_root / "thermal" / f"img{frame_num:05d}_m{m:03d}.npy"
            save_patch(rp, s["rgb_patch"])
            save_patch(tp, s["th8_patch"])
            records.append({
                "image_index": frame_num, "module_index": m,
                "rgb_npy": str(rp), "thermal_npy": str(tp),
                "score": float(s["score"]),
                "area": int(s["area"]), "sat_z": float(s["sat_z"]),
                "t95": float(s["t95"]),
                "n_accepted_panels": s["n_accepted_panels"],
                "preprocessing_version": PREPROCESSING_VERSION,
            })

    idx = pd.DataFrame(records)
    if len(idx) > 0:
        # NOTE: no label is assigned here. Assigning a global quantile
        # threshold at this point would let validation/test score
        # distributions influence the label boundary (reviewer-identified
        # as avoidable test-distribution leakage, even though this is
        # pseudo-labeling rather than ordinary supervised ground truth).
        # Labels are assigned AFTER the leakage-safe split, using a
        # threshold derived from TRAINING scores only, then frozen and
        # applied unchanged to validation/test. See assign_frozen_labels().
        idx.to_csv(out_root / "index_unlabeled.csv", index=False)
    print(f"Built {len(idx)} crops from {idx['image_index'].nunique() if len(idx) else 0} frames")
    print(f"Excluded frames (all reasons): {len(excluded)}")
    if excluded:
        excluded_df = pd.DataFrame(excluded)
        print("Excluded frames by reason:")
        print(excluded_df["reason"].value_counts().to_string())
    print(f"Failed frames (load/process error): {len(failed)}")
    if excluded:
        excluded_df.to_csv(out_root / "excluded_frames.csv", index=False)
        print(f"  -> logged to {out_root / 'excluded_frames.csv'}")
    if failed:
        pd.DataFrame(failed).to_csv(out_root / "failed_frames.csv", index=False)
        print(f"  -> logged to {out_root / 'failed_frames.csv'}")
    return idx


def safe_reset_audit_dir(out_dir: Path, force: bool = False):
    """
    Use this INSTEAD OF raw shutil.rmtree(out_dir) before calling
    dump_candidates_for_review(). Refuses to delete a directory that shows
    signs of completed manual review work (a non-empty panel/, non_panel/,
    or uncertain/ subfolder, or a saved labeled_filter_evaluation.csv),
    unless force=True is explicitly passed.

    This exists because a prior version of the setup instructions had users
    call shutil.rmtree(out_dir) directly with no such check, which silently
    destroyed a completed manual sort. That mistake should not be repeatable.
    """
    import shutil

    out_dir = Path(out_dir)
    if not out_dir.exists():
        return

    signs_of_work = []
    for sub in ("panel", "non_panel", "uncertain"):
        subdir = out_dir / sub
        if subdir.exists() and any(subdir.iterdir()):
            signs_of_work.append(f"{sub}/ contains {len(list(subdir.iterdir()))} file(s)")
    if (out_dir / "labeled_filter_evaluation.csv").exists():
        signs_of_work.append("labeled_filter_evaluation.csv exists")

    if signs_of_work and not force:
        raise RuntimeError(
            f"Refusing to delete {out_dir} - it appears to contain completed "
            f"review work: {'; '.join(signs_of_work)}. If you are certain you "
            f"want to discard this and start over, call "
            f"safe_reset_audit_dir({out_dir!r}, force=True) explicitly. "
            f"Otherwise, use a new directory name instead."
        )

    shutil.rmtree(out_dir)
    print(f"Removed {out_dir} (force={force}, prior signs of work: {signs_of_work or 'none'})")


def dump_candidates_for_review(df, load_rgb_fn, thermogram_cls, make_aligned_pair_fn,
                                out_dir: Path,
                                n_clean_frames=80, seed=42,
                                known_bad_ranges=((0, 32), (2518, 2540))):
    """
    Builds a manual-review sample: saves every candidate blob's RGB crop
    plus its sat_z score, so you can label each as true-panel or non-panel
    and compute real precision/recall for SAT_Z_THRESHOLD - rather than
    treating it as validated from a handful of hand-picked examples.

    IMPORTANT: a purely random sample would almost certainly miss the known
    contaminated frames entirely (56 out of 2541 - under 1 expected in a
    random sample of 40). This function ALWAYS includes every frame in
    known_bad_ranges, plus a random sample of ordinary frames, so the
    review set actually tests whether the filter catches what it's meant to.

    After manually sorting the saved crops into two folders (panel/, non_panel/),
    use `score_saturation_filter()` below to get the actual precision/recall.
    """
    df = df.copy()
    df["frame_num"] = df["rgb_path"].apply(get_frame_num)

    known_bad_mask = df["frame_num"].apply(
        lambda f: any(lo <= f <= hi for lo, hi in known_bad_ranges)
    )
    known_bad = df[known_bad_mask]
    clean_pool = df[~known_bad_mask]
    clean_sample = clean_pool.sample(n=min(n_clean_frames, len(clean_pool)), random_state=seed)
    sample_rows = pd.concat([known_bad, clean_sample]).drop_duplicates()
    print(f"Review set: {len(known_bad)} known-contaminated frames + "
          f"{len(clean_sample)} random clean frames = {len(sample_rows)} total")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for _, row in sample_rows.iterrows():
        frame_num = row["frame_num"]
        try:
            rgb = load_rgb_fn(Path(row["rgb_path"]))
            th = thermogram_cls(Path(row["thermal_path"]))
            th8 = th.render8()
            rgb_crop, th8_crop = make_aligned_pair_fn(rgb, th8)
        except Exception:
            continue

        mask01 = postprocess_mask(preprocess_rgb_hsv_otsu(rgb_crop))
        n, labels = cv2.connectedComponents(mask01)
        hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2HSV)
        cands = []
        for lab in range(1, n):
            blob_mask = labels == lab
            area = int(blob_mask.sum())
            if area < 500:
                continue
            bb = bbox_from_label(labels, lab)
            if bb is None:
                continue
            med_sat = float(np.median(hsv[:, :, 1][blob_mask]))
            cands.append({"bbox": bb, "med_sat": med_sat})
        if not cands:
            continue
        sats = np.array([c["med_sat"] for c in cands])
        med, mad = np.median(sats), robust_mad(sats)
        for i, c in enumerate(cands):
            sat_z = (c["med_sat"] - med) / mad
            x1, y1, x2, y2 = c["bbox"]
            crop = rgb_crop[y1:y2, x1:x2]
            fname = f"frame{frame_num:05d}_cand{i:02d}.png"
            cv2.imwrite(str(out_dir / fname), crop)
            manifest.append({"frame_num": frame_num, "cand_idx": i,
                              "sat_z": sat_z, "filename": fname})

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Dumped {len(manifest)} candidates from {len(sample_rows)} frames to {out_dir}")
    print("Manually sort each image by eye, then run score_saturation_filter().")


def classify_scene_type(frame_num: int,
                         checkerboard_range=(0, 32),
                         vehicle_person_range=(2518, 2540)) -> str:
    """Categorizes a frame by known scene type, for breaking down filter
    performance separately per category (a filter can perform very
    differently on white vehicles vs. black/white checkerboard targets)."""
    if checkerboard_range[0] <= frame_num <= checkerboard_range[1]:
        return "checkerboard_sequence"
    if vehicle_person_range[0] <= frame_num <= vehicle_person_range[1]:
        return "vehicle_person_sequence"
    return "ordinary_flight"


def score_saturation_filter(out_dir: Path, panel_subdir="panel", non_panel_subdir="non_panel",
                             uncertain_subdir="uncertain",
                             threshold=SAT_Z_THRESHOLD, by_scene_type=True):
    """
    After manually sorting dump_candidates_for_review()'s output into
    <out_dir>/panel/ and <out_dir>/non_panel/ subfolders (and optionally
    <out_dir>/uncertain/ for genuinely ambiguous cases), run this to get
    real precision/recall for the saturation threshold.

    Every candidate in manifest.csv must be sorted into one of the three
    folders - this function asserts that, rather than silently dropping
    unlabeled files (which would make the filter look artificially better
    by excluding the hard cases from the reported metrics).

    If by_scene_type=True, also reports the confusion matrix separately for
    each known scene type (checkerboard sequence, vehicle/person sequence,
    ordinary flight) - a filter can perform very differently across these,
    and an aggregate number alone can hide that.
    """
    out_dir = Path(out_dir)
    manifest = pd.read_csv(out_dir / "manifest.csv")
    panel_files = {f.name for f in (out_dir / panel_subdir).glob("*.png")}
    non_panel_files = {f.name for f in (out_dir / non_panel_subdir).glob("*.png")}
    uncertain_dir = out_dir / uncertain_subdir
    uncertain_files = {f.name for f in uncertain_dir.glob("*.png")} if uncertain_dir.exists() else set()

    def _label(f):
        if f in panel_files: return "panel"
        if f in non_panel_files: return "non_panel"
        if f in uncertain_files: return "uncertain"
        return None

    manifest["true_label"] = manifest["filename"].apply(_label)

    n_unlabeled = manifest["true_label"].isna().sum()
    assert n_unlabeled == 0, (
        f"{n_unlabeled} candidate images have not been manually sorted into "
        f"{panel_subdir}/, {non_panel_subdir}/, or {uncertain_subdir}/. "
        f"Sort every candidate before scoring - do not proceed with a partial set."
    )

    n_uncertain = (manifest["true_label"] == "uncertain").sum()
    if n_uncertain > 0:
        print(f"[note] {n_uncertain} candidates marked 'uncertain' - excluded from "
              f"precision/recall below but reported separately, not silently dropped.")

    manifest["predicted_accept"] = manifest["sat_z"] >= threshold
    if by_scene_type:
        manifest["scene_type"] = manifest["frame_num"].apply(classify_scene_type)

    manifest.to_csv(out_dir / "labeled_filter_evaluation.csv", index=False)
    print(f"Saved full candidate-level evaluation to {out_dir / 'labeled_filter_evaluation.csv'}")

    labeled = manifest[manifest["true_label"] != "uncertain"].copy()

    def _confusion(df):
        tp = ((df["true_label"] == "panel") & df["predicted_accept"]).sum()
        fp = ((df["true_label"] == "non_panel") & df["predicted_accept"]).sum()
        fn = ((df["true_label"] == "panel") & ~df["predicted_accept"]).sum()
        tn = ((df["true_label"] == "non_panel") & ~df["predicted_accept"]).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        non_panel_rejection_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {"n": len(df), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "panel_recall": recall, "precision": precision,
                "non_panel_rejection_rate": non_panel_rejection_rate}

    overall = _confusion(labeled)
    print(f"\n=== OVERALL (n={overall['n']}, excludes {n_uncertain} uncertain) ===")
    print(f"  TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} TN={overall['tn']}")
    print(f"  Panel recall (of real panels, % correctly accepted):        {overall['panel_recall']:.3f}")
    print(f"  Precision (of accepted, % that are real panels):            {overall['precision']:.3f}")
    print(f"  Non-panel rejection rate (of non-panels, % correctly rejected): {overall['non_panel_rejection_rate']:.3f}")
    print(
        "\n[interpretation note] Ordinary-flight frames contain many genuine panels and "
        "few non-panels, so OVERALL precision can look high even if several cars/markers "
        "slipped through. Weight your judgment toward the non-panel rejection rate and "
        "raw FP/FN counts in the checkerboard/vehicle_person breakdowns below, not the "
        "aggregate precision number alone."
    )

    results = {"overall": overall, "n_uncertain": int(n_uncertain)}
    if by_scene_type:
        print()
        for scene in labeled["scene_type"].unique():
            sub = labeled[labeled["scene_type"] == scene]
            if len(sub) == 0:
                continue
            r = _confusion(sub)
            results[scene] = r
            print(f"=== {scene} (n={r['n']}) ===")
            print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}")
            print(f"  Panel recall: {r['panel_recall']:.3f} | Precision: {r['precision']:.3f} | "
                  f"Non-panel rejection: {r['non_panel_rejection_rate']:.3f}")

    return results


def assign_frozen_labels(index_df: pd.DataFrame, out_root: Path,
                          top_frac: float = 0.12) -> pd.DataFrame:
    """
    Assigns labels using a threshold derived ONLY from training-split
    scores, then freezes and applies that same threshold to validation
    and test - avoiding the test-distribution leakage of a global
    quantile computed over the whole (already-split) dataset.

    index_df must already have a 'split' column (train/val/test) from a
    leakage-safe split built on UNLABELED scores (see the module-level
    note about why rate-balancing by label is incompatible with this
    protocol - use a volume-only split, not balance_split_by_rate).

    Saves index_labeled.csv and label_threshold.json (full provenance:
    threshold value, method, preprocessing version) to out_root.
    """
    import json

    train_mask = index_df["split"].eq("train")
    if train_mask.sum() == 0:
        raise ValueError("No training-split rows found -- run the split before labeling.")

    label_threshold = float(index_df.loc[train_mask, "score"].quantile(1 - top_frac))
    index_df = index_df.copy()
    index_df["label"] = (index_df["score"] >= label_threshold).astype("int8")
    index_df["label_threshold"] = label_threshold
    index_df["label_threshold_source"] = "train_score_quantile_%.2f" % (1 - top_frac)

    out_root = Path(out_root)
    index_df.to_csv(out_root / "index_labeled.csv", index=False)

    threshold_record = {
        "score_definition": "panel_mask_T95_relative_to_frame_panel_median_MAD",
        "threshold_method": "training-derived quantile",
        "training_quantile": 1 - top_frac,
        "target_training_anomaly_fraction": top_frac,
        "threshold": label_threshold,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    with open(out_root / "label_threshold.json", "w") as f:
        json.dump(threshold_record, f, indent=2)

    print(f"Frozen training-derived threshold: {label_threshold:.6f}")
    print("Class prevalence by split (NOT forced equal - this is honest, not a bug):")
    print(index_df.groupby("split")["label"].agg(["count", "sum", "mean"]).to_string())
    return index_df
