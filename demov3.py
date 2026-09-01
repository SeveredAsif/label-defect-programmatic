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

V3_DIR = Path(os.environ.get("LABEL_V3_DIR", "v3_evaluation_REPORT"))
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

# Per-sample cap on how many golden/faulty cases get evaluated, so a sample
# with abundant real photos (e.g. sample1's 44 real faulty images) doesn't
# blow past a sample that only has the augmented top-up (e.g. 20). Set to
# None to disable and evaluate every available image, as before. Matches
# the targets `generate_augmented_dataset.py` tops each sample up to, so
# together they produce a controlled, predictable total
# (5 samples x (40 golden + 20 faulty) = 300 evaluations, 100 golden-vs-
# faulty comparisons).
MAX_GOLDEN_PER_SAMPLE = int(os.environ.get("LABEL_MAX_GOLDEN_PER_SAMPLE", "40"))
MAX_FAULTY_PER_SAMPLE = int(os.environ.get("LABEL_MAX_FAULTY_PER_SAMPLE", "20"))


def _cap(paths, max_count, sample_name, kind):
    if max_count is None or len(paths) <= max_count:
        return paths
    rng = np.random.default_rng(evaluation.stable_seed(f"{sample_name}_{kind}_cap"))
    idx = rng.choice(len(paths), size=max_count, replace=False)
    idx.sort()
    kept = [paths[i] for i in idx]
    print(f"{sample_name}: capping {kind} from {len(paths)} to {max_count} (deterministic random subset)")
    return kept


def _is_usable_crop(path):
    img = cv2.imread(str(path))
    if img is None:
        return False
    h, w = img.shape[:2]
    return h >= MIN_CROP_DIM and w >= MIN_CROP_DIM


def collect_sample_dirs():
    return [d for d in sorted(evaluation.DATASET_ROOT.iterdir()) if d.is_dir()]


def find_faulty_dir(sample_dir):
    """The sample's real-faulty-photos folder (named `faulty` or
    `*_faulty`) — kept separate from the `faulty_augmented/<technique>/`
    folders generate_augmented_dataset.py writes to, so real and augmented
    images are never mixed in the same directory."""
    for d in sorted(sample_dir.iterdir()):
        if d.is_dir() and (d.name.lower().endswith("_faulty") or d.name.lower() == "faulty"):
            return d
    return None


def _collect_usable(folder):
    """All usable image files directly inside `folder`, recursing into
    subfolders (e.g. `golden_augmented/<technique>/...`) — degenerate/
    unreadable crops are skipped with a note."""
    if not folder.exists():
        return []
    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in evaluation.EXTENSIONS
    )
    usable = [p for p in paths if _is_usable_crop(p)]
    for p in paths:
        if p not in usable:
            print(f"Skipping degenerate crop (too small / unreadable): {p}")
    return usable


def load_golden_paths(sample_dir):
    """Real goldens (`golden/`) plus any augmented golden variants
    (`golden_augmented/<technique>/`, produced by
    generate_augmented_dataset.py — never mixed into `golden/` itself)."""
    paths = _collect_usable(sample_dir / "golden")
    paths += _collect_usable(sample_dir / "golden_augmented")
    return sorted(paths)


def load_faulty_cases(sample_dir):
    """Real faulty photos plus any augmented faulty variants
    (`faulty_augmented/<technique>/`, produced by
    generate_augmented_dataset.py — never mixed into the real faulty
    folder itself)."""
    faulty_dir = find_faulty_dir(sample_dir)
    cases = _collect_usable(faulty_dir) if faulty_dir is not None else []
    cases += _collect_usable(sample_dir / "faulty_augmented")
    return sorted(cases)


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
        if len(golden_paths) == 0:
            print(f"Skipping {sample_dir.name}: no usable golden images found.")
            continue
        golden_paths = _cap(golden_paths, MAX_GOLDEN_PER_SAMPLE, sample_dir.name, "golden")
        faulty_paths = _cap(load_faulty_cases(sample_dir), MAX_FAULTY_PER_SAMPLE, sample_dir.name, "faulty")

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
        if len(golden_paths) >= 2:
            # Leave-one-out: each golden is tested as a "clean" candidate
            # against the *rest* of the sample's real goldens.
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
        else:
            # Only one golden in this sample: there's nothing to hold it out
            # against (leave-one-out needs >=2), so there's no meaningful
            # "clean" self-consistency case to generate here — comparing the
            # lone golden to itself would trivially always pass and wouldn't
            # tell us anything. Faulty cases below still get tested normally
            # against this single golden.
            print(
                f"{sample_dir.name}: only 1 golden found; skipping leave-one-out "
                f"clean-case test (nothing to hold out against). Faulty cases "
                f"will still be tested against it."
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

            g2 = report.gate2
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
                    "ssim_score": f"{g2.ssim_score:.6f}" if g2 and g2.ssim_score is not None else "",
                    "num_hotspots": len(g2.hotspots) if g2 is not None else 0,
                    "gate1_passed": report.gate1.passed,
                    "gate2_passed": g2.passed if g2 is not None else "",
                    "num_good_matches": g2.num_good_matches if g2 is not None else "",
                    "severity_frac": f"{g2.severity_frac:.6f}" if g2 and g2.severity_frac is not None else "",
                    "diff_severity_frac": f"{g2.diff_severity_frac:.6f}" if g2 and g2.diff_severity_frac is not None else "",
                    "max_hotspot_area_frac": f"{g2.max_hotspot_area_frac:.6f}" if g2 and g2.max_hotspot_area_frac is not None else "",
                    "hotspot_area_frac": f"{g2.hotspot_area_frac:.6f}" if g2 and g2.hotspot_area_frac is not None else "",
                    "reasons": " | ".join(g2.reasons) if g2 is not None else "",
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
        "gate1_passed", "gate2_passed", "num_good_matches", "severity_frac",
        "diff_severity_frac", "max_hotspot_area_frac", "hotspot_area_frac", "reasons",
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
