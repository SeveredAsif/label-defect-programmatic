"""
Evaluate the Kabir crops with the V2 multi-golden ensemble pipeline
(`pipelinev2.LabelInspectorV2`).

This is the ensemble-aware counterpart to `demov3.py`. demov3 exposed the
core problem: comparing real candidates against a *single* real golden
photo produces a 94% false-positive rate, because natural unit-to-unit
variation between real golden samples (fabric/print tolerance, handheld
lighting, camera noise) is misread as a defect. Here, every sample folder's
full set of real golden crops is used as an ensemble reference (with
per-sample threshold calibration derived from the goldens' own natural
variation), instead of picking just the first crop as "the" golden.

Clean cases are evaluated leave-one-out: each real golden crop is tested as
a candidate against the *rest* of that sample's goldens (never against
itself), so the evaluation stays honest. Faulty crops are tested against
the full golden ensemble for that sample.

Reuses demov2.py's I/O/report/scoring helpers so v2/v3/v4 stay directly
comparable.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LABEL_DATASET_ROOT", "cropped_kabir_golden_faulty_folderwise")

import cv2
import numpy as np

import demov2 as evaluation
from pipelinev2 import LabelInspectorV2

V4_DIR = Path(os.environ.get("LABEL_V4_DIR", "v4_evaluation"))
evaluation.OUT_DIR = V4_DIR
evaluation.EVAL_DIR = V4_DIR
evaluation.ALL_REPORTS_DIR = V4_DIR / "all_case_reports"
evaluation.ANALYSIS_DIR = V4_DIR / "analysis_images"
evaluation.AUGMENTED_DIR = V4_DIR / "augmented_goldens"  # unused, kept for dir compatibility
for directory in (
    evaluation.EVAL_DIR,
    evaluation.ALL_REPORTS_DIR,
    evaluation.ANALYSIS_DIR,
    evaluation.AUGMENTED_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

evaluation.REPORT_PATH = V4_DIR / "rsn_v4.txt"
evaluation.SUMMARY_PATH = V4_DIR / "rsn_v4.txt"
evaluation.PREDICTIONS_PATH = V4_DIR / "predictions_v4.csv"
evaluation.CLASSIFICATION_REPORT_PATH = V4_DIR / "classification_report_v4.csv"
evaluation.HARD_MISTAKES_PATH = V4_DIR / "hardest_mistakes_v4.csv"

# Five evenly-distributed real references (odd, so the median pick in
# inspect_ensemble is a clear single middle value) retain the useful
# natural variation while keeping batch evaluation practical on 259 crop
# cases. Was 3 — bumped since the median-of-ensemble selection (see
# inspect_ensemble) benefits from a few more samples to be robust to any
# single geometrically-lucky/unlucky pairing.
MAX_ENSEMBLE_SIZE = 5
MAX_CALIBRATION_PAIRS = 5


def collect_sample_dirs():
    return [d for d in sorted(evaluation.DATASET_ROOT.iterdir()) if d.is_dir()]


MIN_CROP_DIM = 20  # px; smaller than this is a cropping-script artifact, not a real label photo


def _is_usable_crop(path):
    img = cv2.imread(str(path))
    if img is None:
        return False
    h, w = img.shape[:2]
    return h >= MIN_CROP_DIM and w >= MIN_CROP_DIM


def load_golden_paths(sample_dir):
    paths = sorted(
        p for p in (sample_dir / "golden").iterdir()
        if p.is_file() and p.suffix.lower() in evaluation.EXTENSIONS
    )
    usable = [p for p in paths if _is_usable_crop(p)]
    for p in paths:
        if p not in usable:
            print(f"Skipping degenerate crop (too small / unreadable): {p}")
    return usable


def load_faulty_cases(sample_dir):
    cases = []
    faulty_dirs = [
        d for d in sorted(sample_dir.iterdir())
        if d.is_dir() and (d.name.lower().endswith("_faulty") or d.name.lower() == "faulty")
    ]
    for faulty_dir in faulty_dirs:
        for p in sorted(faulty_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in evaluation.EXTENSIONS:
                if _is_usable_crop(p):
                    cases.append(p)
                else:
                    print(f"Skipping degenerate crop (too small / unreadable): {p}")
    return cases


def estimate_target_size_ensemble(golden_bgr_list, sample_n=5):
    """Median Gate-1 target size over a few golden crops, so the size gate
    isn't hostage to one particular photo's framing noise."""
    sizes = []
    for g in golden_bgr_list[:sample_n]:
        sizes.append(evaluation.estimate_target_size(g))
    widths, heights = zip(*sizes)
    return (float(np.median(widths)), float(np.median(heights)))


def main():
    sample_dirs = collect_sample_dirs()
    if not sample_dirs:
        raise RuntimeError(f"No sample folders found under: {evaluation.DATASET_ROOT}")

    from collections import defaultdict
    chunks = []
    records = []
    category_examples = defaultdict(int)

    for sample_dir in sample_dirs:
        golden_paths = load_golden_paths(sample_dir)
        if not golden_paths:
            continue
        faulty_paths = load_faulty_cases(sample_dir)

        golden_imgs = {p: evaluation.load(p) for p in golden_paths}
        target_size = estimate_target_size_ensemble(list(golden_imgs.values()))

        inspector = LabelInspectorV2(
            target_size_px=target_size,
            # These images are already SAM/YOLO crops, so crop framing (and
            # the rectangle/polygon bounding box angle) is not evidence of a
            # physical slanted-cut fault.  Keep contour extraction for the
            # mask/diagnostics, but do not reject a crop on its outer-frame
            # size or skew.  Content defects are decided by the aligned
            # contour-aware Gate 2 below.
            # max_skew_deg is now meaningful again: RobustStructuralGate
            # skips the skew check on tight, margin-less crops (where the
            # reading is crop-boundary noise, not a real slant) instead of
            # needing it globally disabled via 90.0 here.
            gate1_kwargs=dict(size_tolerance_pct=10.0, max_skew_deg=6.0),
            gate2_kwargs=dict(
                ssim_defect_threshold=0.55,
                diff_noise_floor=28,
                min_hotspot_area=25,
                overall_ssim_reject_threshold=0.80,
            ),
            max_ensemble_size=MAX_ENSEMBLE_SIZE,
        )
        cal = inspector.calibrate(
            list(golden_imgs.values()), max_pairs=MAX_CALIBRATION_PAIRS
        )
        if cal is not None:
            print(
                f"[{sample_dir.name}] calibrated thresholds from {cal.n_pairs} golden pairs: "
                f"severity>={cal.severity_frac_reject_threshold:.3f} "
                f"diff_severity>={cal.diff_severity_frac_reject_threshold:.3f} "
                f"max_hotspot>={cal.max_hotspot_frac_reject_threshold:.3f} "
                f"ssim<{cal.overall_ssim_reject_threshold:.3f}"
            )

        cases = []
        for golden_path in golden_paths:
            ensemble_paths = [p for p in golden_paths if p != golden_path]
            cases.append(
                {
                    "set_name": f"{evaluation.slugify(sample_dir.name)}__golden_{golden_path.stem}",
                    "expected": "clean",
                    "source_kind": "golden",
                    "candidate_path": golden_path,
                    "ensemble_imgs": [golden_imgs[p] for p in ensemble_paths],
                    "ensemble_names": [p.name for p in ensemble_paths],
                }
            )
        for faulty_path in faulty_paths:
            cases.append(
                {
                    "set_name": f"{evaluation.slugify(sample_dir.name)}__{faulty_path.stem}",
                    "expected": "faulty",
                    "source_kind": "faulty",
                    "candidate_path": faulty_path,
                    "ensemble_imgs": list(golden_imgs.values()),
                    "ensemble_names": [p.name for p in golden_imgs.keys()],
                }
            )

        for item in cases:
            set_name = item["set_name"]
            candidate = evaluation.load(item["candidate_path"])
            report = inspector.inspect(
                item["ensemble_imgs"], candidate, golden_names=item["ensemble_names"]
            )

            outcome = evaluation.evaluate_outcome(item["expected"], report)
            stage = evaluation.stage_label(report)
            case_prefix = f"{outcome}_{stage.replace(' ', '')}_{evaluation.slugify(set_name)}"
            all_case_path = evaluation.ALL_REPORTS_DIR / f"{case_prefix}_report.png"

            mirror_paths = []
            if category_examples[outcome] < 2:
                category_dir = evaluation.build_case_output_dirs(outcome)
                mirror_paths.append(category_dir / f"{case_prefix}_report.png")
                category_examples[outcome] += 1

            # Show the golden actually matched/aligned against, not an
            # arbitrary ensemble member — using ensemble_imgs[0] here used to
            # create misleading panels (mismatched orientation/framing vs.
            # the golden the real comparison used), which read as a
            # "zoomed"/distorted alignment that wasn't actually there.
            golden_for_plot = getattr(report, "matched_golden_bgr", None)
            golden_name_for_plot = getattr(report, "matched_golden_name", None)
            if golden_for_plot is None:
                golden_for_plot = item["ensemble_imgs"][0] if item["ensemble_imgs"] else candidate
                golden_name_for_plot = item["ensemble_names"][0] if item["ensemble_names"] else None
            out_path = evaluation.visualize(
                set_name, golden_for_plot, candidate, report, all_case_path, mirror_paths=mirror_paths,
                golden_label=golden_name_for_plot, candidate_label=item["candidate_path"].name,
            )

            records.append(
                {
                    "case_name": set_name,
                    "expected": item["expected"],
                    "predicted": evaluation.prediction_from_report(report),
                    "outcome": outcome,
                    "stage": stage,
                    "source_kind": item["source_kind"],
                    "brand": sample_dir.name,
                    "golden": f"ensemble(n={len(item['ensemble_imgs'])})",
                    "candidate": str(item["candidate_path"]),
                    "report_path": out_path,
                    "ssim_score": f"{report.gate2.ssim_score:.6f}" if report.gate2 and report.gate2.ssim_score is not None else "",
                    "num_hotspots": len(report.gate2.hotspots) if report.gate2 is not None else 0,
                    "gate1_passed": report.gate1.passed,
                    "gate2_passed": report.gate2.passed if report.gate2 is not None else "",
                }
            )

            print("=" * 70)
            print(set_name)
            print("=" * 70)
            print(report.summary())
            print()

            chunk = [
                "=" * 70,
                set_name,
                "=" * 70,
                report.summary(),
                f"expected: {item['expected']} | predicted: {evaluation.prediction_from_report(report)} | "
                f"outcome: {outcome} | stage: {stage}",
                f"source: ensemble_n={len(item['ensemble_imgs'])} | candidate={item['candidate_path']}",
                "",
                f"Saved visualization -> {out_path}",
                "",
            ]
            chunks.append("\n".join(chunk))

    summary_text, summary_counts = evaluation.summarize_records(records)
    summary_body = "\n\n".join([summary_text] + chunks)

    evaluation.REPORT_PATH.write_text(summary_body, encoding="utf-8")
    evaluation.SUMMARY_PATH.write_text(summary_body, encoding="utf-8")

    fieldnames = [
        "case_name", "expected", "predicted", "outcome", "stage", "source_kind",
        "brand", "golden", "candidate", "report_path", "ssim_score", "num_hotspots",
        "gate1_passed", "gate2_passed",
    ]
    evaluation.write_csv(evaluation.PREDICTIONS_PATH, fieldnames, records)
    evaluation.write_csv(
        evaluation.CLASSIFICATION_REPORT_PATH,
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in summary_counts.items()],
    )

    mistakes = [record for record in records if record["outcome"] in {"FP", "FN"}]
    mistakes.sort(key=evaluation.mistake_score, reverse=True)
    evaluation.write_csv(evaluation.HARD_MISTAKES_PATH, fieldnames, mistakes)

    print(summary_text)
    print()
    print(f"Consolidated report saved -> {evaluation.REPORT_PATH}")
    print(f"Predictions CSV saved -> {evaluation.PREDICTIONS_PATH}")
    print(f"Classification summary saved -> {evaluation.CLASSIFICATION_REPORT_PATH}")
    print(f"Mistakes report saved -> {evaluation.HARD_MISTAKES_PATH}")


if __name__ == "__main__":
    main()
