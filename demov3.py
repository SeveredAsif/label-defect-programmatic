"""Best-match-against-real-goldens evaluation for the Kabir dataset
(v1 `pipeline.LabelInspector`, no ensemble/calibration).

Unlike the old demov3 (which delegated straight to demov2's single-golden +
synthetic-augmentation `main()`), this version:

  - Uses ALL real golden crops in a sample folder as individual reference
    points, not just the first one, and not any synthetic augmentation.
  - For clean cases, tests each golden leave-one-out against the *rest* of
    that sample's real goldens (never against itself).
  - For faulty cases, tests the crop against every real golden in the
    sample.
  - In both cases, a candidate is compared against each golden ONE AT A
    TIME with the plain v1 `LabelInspector` (no multi-golden ensemble
    statistics, no per-sample calibration a la pipelinev2/demov4) — the
    candidate is judged clean if it matches ANY single golden well enough
    (nearest-neighbor style), and the best (most lenient) of those
    per-golden reports is what gets recorded/visualized. If it fails to
    match every golden, the least-bad rejection is recorded/visualized.

Results are written to ``v3_evaluation``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LABEL_DATASET_ROOT", "cropped_kabir_golden_faulty_folderwise")

import cv2
import numpy as np

import demov2 as evaluation
from pipeline import LabelInspector

V3_DIR = Path(os.environ.get("LABEL_V3_DIR", "v3_evaluation_final"))
evaluation.OUT_DIR = V3_DIR
evaluation.EVAL_DIR = V3_DIR
evaluation.ALL_REPORTS_DIR = V3_DIR / "all_case_reports"
evaluation.ANALYSIS_DIR = V3_DIR / "analysis_images"
evaluation.AUGMENTED_DIR = V3_DIR / "augmented_goldens"  # unused, kept for dir compatibility
for directory in (
    evaluation.EVAL_DIR,
    evaluation.ALL_REPORTS_DIR,
    evaluation.ANALYSIS_DIR,
    evaluation.AUGMENTED_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

evaluation.REPORT_PATH = V3_DIR / "rsn_v3.txt"
evaluation.SUMMARY_PATH = V3_DIR / "rsn_v3.txt"
evaluation.PREDICTIONS_PATH = V3_DIR / "predictions_v3.csv"
evaluation.CLASSIFICATION_REPORT_PATH = V3_DIR / "classification_report_v3.csv"
evaluation.HARD_MISTAKES_PATH = V3_DIR / "hardest_mistakes_v3.csv"

MIN_CROP_DIM = 20  # px; smaller than this is a cropping-script artifact, not a real label photo


def _is_usable_crop(path):
    img = cv2.imread(str(path))
    if img is None:
        return False
    h, w = img.shape[:2]
    return h >= MIN_CROP_DIM and w >= MIN_CROP_DIM


def collect_sample_dirs():
    return [d for d in sorted(evaluation.DATASET_ROOT.iterdir()) if d.is_dir()]


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


def estimate_target_size_median(golden_bgr_list):
    """Median Gate-1 target size over all golden crops in the sample, so the
    size gate isn't hostage to one particular photo's framing noise."""
    sizes = [evaluation.estimate_target_size(g) for g in golden_bgr_list]
    widths, heights = zip(*sizes)
    return (float(np.median(widths)), float(np.median(heights)))


def _report_rank_key(report):
    """Lower is better/more-lenient. PASS beats REJECT_GATE2 beats
    REJECT_GATE1; within PASS/REJECT_GATE2, rank by how mild the content
    mismatch was so the single best-matching golden wins."""
    if report.verdict == "PASS":
        severity = report.gate2.severity_frac if report.gate2 is not None else 0.0
        return (0, severity)
    if report.verdict == "REJECT_GATE2":
        severity = report.gate2.severity_frac if report.gate2 is not None else 1.0
        return (1, severity)
    return (2, 0.0)  # REJECT_GATE1


def best_match_report(inspector, golden_bgr_list, candidate_bgr):
    """Compare candidate against every golden individually and keep the
    single best (most lenient) result — nearest-neighbor style, no
    ensemble statistics."""
    reports = [(g, inspector.inspect(g, candidate_bgr)) for g in golden_bgr_list]
    best_golden, best_report = min(reports, key=lambda pair: _report_rank_key(pair[1]))
    return best_report, best_golden


def main():
    sample_dirs = collect_sample_dirs()
    if not sample_dirs:
        raise RuntimeError(f"No sample folders found under: {evaluation.DATASET_ROOT}")

    chunks = []
    records = []
    category_examples = defaultdict(int)

    for sample_dir in sample_dirs:
        golden_paths = load_golden_paths(sample_dir)
        if len(golden_paths) < 2:
            print(f"Skipping {sample_dir.name}: need >=2 real goldens for leave-one-out, found {len(golden_paths)}")
            continue
        faulty_paths = load_faulty_cases(sample_dir)

        golden_imgs = {p: evaluation.load(p) for p in golden_paths}
        target_size = estimate_target_size_median(list(golden_imgs.values()))

        inspector = LabelInspector(
            target_size_px=target_size,
            gate1_kwargs=dict(size_tolerance_pct=0.35, max_skew_deg=6.0),
            gate2_kwargs=dict(
                ssim_defect_threshold=0.55,
                diff_noise_floor=28,
                min_hotspot_area=25,
                overall_ssim_reject_threshold=0.80,
            ),
        )

        cases = []
        for golden_path in golden_paths:
            reference_paths = [p for p in golden_paths if p != golden_path]
            cases.append(
                {
                    "set_name": f"{evaluation.slugify(sample_dir.name)}__golden_{golden_path.stem}",
                    "expected": "clean",
                    "source_kind": "golden",
                    "candidate_path": golden_path,
                    "reference_imgs": [golden_imgs[p] for p in reference_paths],
                }
            )
        for faulty_path in faulty_paths:
            cases.append(
                {
                    "set_name": f"{evaluation.slugify(sample_dir.name)}__{faulty_path.stem}",
                    "expected": "faulty",
                    "source_kind": "faulty",
                    "candidate_path": faulty_path,
                    "reference_imgs": list(golden_imgs.values()),
                }
            )

        for item in cases:
            set_name = item["set_name"]
            candidate = evaluation.load(item["candidate_path"])
            report, matched_golden = best_match_report(inspector, item["reference_imgs"], candidate)

            outcome = evaluation.evaluate_outcome(item["expected"], report)
            stage = evaluation.stage_label(report)
            case_prefix = f"{outcome}_{stage.replace(' ', '')}_{evaluation.slugify(set_name)}"
            all_case_path = evaluation.ALL_REPORTS_DIR / f"{case_prefix}_report.png"

            mirror_paths = []
            if category_examples[outcome] < 2:
                category_dir = evaluation.build_case_output_dirs(outcome)
                mirror_paths.append(category_dir / f"{case_prefix}_report.png")
                category_examples[outcome] += 1

            out_path = evaluation.visualize(
                set_name, matched_golden, candidate, report, all_case_path, mirror_paths=mirror_paths
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
                    "golden": f"best_of(n={len(item['reference_imgs'])})",
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
                f"source: best_of_n={len(item['reference_imgs'])} | candidate={item['candidate_path']}",
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
