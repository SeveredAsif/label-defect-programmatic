"""
Demo: run the two-gate pipeline on the two golden/error image sets supplied
by the user, and save diagnostic visualizations (registration overlay,
SSIM heatmap, diff mask, hot-spot crops) to disk.

Set A: size-chart label
    golden   = WhatsApp_Image_2026-07-02_at_3_05_52_PM__1_.jpeg
    error    = WhatsApp_Image_2026-07-02_at_3_05_52_PM.jpeg

Set B: Bangladesh care-instructions label
    golden   = WhatsApp_Image_2026-07-02_at_3_02_59_PM__1_.jpeg
    error    = WhatsApp_Image_2026-07-02_at_3_02_59_PM.jpeg

NOTE: since these are real phone photos (not a fixed inspection rig), each
pair is first roughly matched in scale before Gate 1's target size check —
in a real production line the camera/backlight rig is fixed, so target_size
would be a constant calibrated once. Here we derive a reasonable target
from the golden sample itself, since we only have single examples.
"""

import csv
import os
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import LabelInspector, StructuralGate

DATASET_ROOT = Path("label_dataset") / "label_dataset"
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = OUT_DIR / "evaluation_results"
ALL_REPORTS_DIR = EVAL_DIR / "all_case_reports"
ANALYSIS_DIR = EVAL_DIR / "analysis_images"
AUGMENTED_DIR = EVAL_DIR / "augmented_goldens"
for directory in (EVAL_DIR, ALL_REPORTS_DIR, ANALYSIS_DIR, AUGMENTED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUT_DIR / "rsn.txt"
SUMMARY_PATH = EVAL_DIR / "rsn.txt"
PREDICTIONS_PATH = EVAL_DIR / "predictions.csv"
CLASSIFICATION_REPORT_PATH = EVAL_DIR / "classification_report.csv"
HARD_MISTAKES_PATH = EVAL_DIR / "hardest_mistakes.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_NAME_MAP = {
    "Primark": "Primark",
    "zara": "Zara",
    "zara_narrow": "ZaraNarrow",
}

OUTCOME_STAGE_LABEL = {
    "REJECT_GATE1": "Gate 1",
    "REJECT_GATE2": "Gate 2",
    "PASS": "PASS",
}


def load(path, max_dim=700):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def slugify(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def stable_seed(text):
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _apply_affine(img, matrix):
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def augment_golden(golden_bgr, seed_text, count=5):
    rng = np.random.default_rng(stable_seed(seed_text))
    h, w = golden_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    augmentations = []

    angle = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.99, 1.01))
    mat = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    mat[0, 2] += rng.uniform(-2, 2)
    mat[1, 2] += rng.uniform(-2, 2)
    augmentations.append(("rotate", _apply_affine(golden_bgr, mat)))

    alpha = float(rng.uniform(0.95, 1.05))
    beta = float(rng.uniform(-8, 8))
    bright = cv2.convertScaleAbs(golden_bgr, alpha=alpha, beta=beta)
    augmentations.append(("brightness", bright))

    blur_k = int(rng.choice([3]))
    blurred = cv2.GaussianBlur(golden_bgr, (blur_k, blur_k), 0)
    noise = rng.normal(0, rng.uniform(1.0, 3.0), golden_bgr.shape).astype(np.int16)
    noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmentations.append(("blur_noise", noisy))

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [rng.uniform(0, 2), rng.uniform(0, 2)],
        [w - rng.uniform(0, 2), rng.uniform(0, 2)],
        [w - rng.uniform(0, 2), h - rng.uniform(0, 2)],
        [rng.uniform(0, 2), h - rng.uniform(0, 2)],
    ])
    dst += rng.uniform(-3, 3, size=(4, 2)).astype(np.float32)
    dst = np.clip(dst, [0, 0], [w - 1, h - 1]).astype(np.float32)
    pm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        golden_bgr,
        pm,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    augmentations.append(("perspective", warped))

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(golden_bgr, -1, sharpen_kernel)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.integers(-6, 7), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.integers(-10, 11), 0, 255)
    color_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmentations.append(("color_shift", color_shifted))

    return augmentations[:count]


def normalize_verdict(report):
    return report.verdict if report.verdict in OUTCOME_STAGE_LABEL else "PASS"


def prediction_from_report(report):
    return "FAULT" if normalize_verdict(report) != "PASS" else "CLEAN"


def evaluate_outcome(expected_label, report):
    predicted_label = prediction_from_report(report)
    if expected_label == "faulty" and predicted_label == "FAULT":
        return "TP"
    if expected_label == "faulty" and predicted_label == "CLEAN":
        return "FN"
    if expected_label == "clean" and predicted_label == "FAULT":
        return "FP"
    return "TN"


def stage_label(report):
    return OUTCOME_STAGE_LABEL.get(normalize_verdict(report), "PASS")


def save_augmented_samples(brand_name, golden_path, golden_bgr):
    brand_dir = AUGMENTED_DIR / slugify(brand_name)
    brand_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, (aug_name, aug_img) in enumerate(augment_golden(golden_bgr, str(golden_path), count=5), start=1):
        file_name = f"{slugify(Path(golden_path).stem)}__{index:02d}_{aug_name}.png"
        out_path = brand_dir / file_name
        cv2.imwrite(str(out_path), aug_img)
        samples.append((aug_name, aug_img, out_path))
    return samples

def estimate_target_size(golden_bgr):
    """
    Estimate target size directly from the golden image dimensions.
    Normalize so width <= height, matching StructuralGate.inspect().
    """
    h, w = golden_bgr.shape[:2]

    width = min(w, h)
    height = max(w, h)
    

    print(f"Golden image size (normalized): {width} x {height}")

    return (width, height)



# def estimate_target_size(golden_bgr):
#     """Derive a Gate-1 target size directly from the golden sample's own
#     minAreaRect, since we only have one reference photo per set (no fixed
#     calibrated rig here). In production, replace with a constant measured
#     from the approved physical spec."""
#     tmp_gate = StructuralGate(target_size_px=(1, 1))  # dummy target, unused here
#     gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     mask = tmp_gate._binarize(gray)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     main = max(contours, key=cv2.contourArea)
#     rect = cv2.minAreaRect(main)
#     w, h = rect[1]
#     print(w,h)
#     return (min(w, h), max(w, h))


def visualize(set_name, golden_bgr, candidate_bgr, report, out_path, mirror_paths=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Label Inspection — {set_name} — verdict: {report.verdict}", fontsize=14)

    def show(ax, img_bgr, title, cmap=None):
        if img_bgr is None:
            ax.set_title(title + " (n/a)")
            ax.axis("off")
            return
        if cmap:
            ax.imshow(img_bgr, cmap=cmap)
        else:
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    show(axes[0, 0], golden_bgr, "Golden Reference")

    cand_annot = candidate_bgr.copy()
    if report.gate1.box_points is not None:
        cv2.drawContours(cand_annot, [np.int32(report.gate1.box_points)], 0, (0, 255, 0), 2)
    show(axes[0, 1], cand_annot, f"Candidate + Gate1 box\nskew={report.gate1.angle_deg:.2f} deg"
         if report.gate1.angle_deg is not None else "Candidate + Gate1 box")

    show(axes[0, 2], report.gate1.mask, "Gate 1 binary mask", cmap="gray")

    if report.gate2 is not None:
        show(axes[1, 0], report.gate2.aligned_candidate, "Aligned candidate (registered)")

        ssim_vis = ((1 - report.gate2.ssim_map) * 255).clip(0, 255).astype(np.uint8)
        show(axes[1, 1], ssim_vis, f"SSIM anomaly map\nscore={report.gate2.ssim_score:.3f}", cmap="inferno")

        annotated = report.gate2.aligned_candidate.copy()
        for hs in report.gate2.hotspots:
            x, y, w, h = hs.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"{hs.defect_class} ({hs.confidence:.2f})" if hs.defect_class else "?"
            cv2.putText(annotated, label, (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        show(axes[1, 2], annotated, f"Hot spots ({len(report.gate2.hotspots)})")
    else:
        for ax in axes[1, :]:
            ax.axis("off")
            ax.set_title("Gate 2 skipped (Gate 1 rejected)")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130)
    for extra_path in mirror_paths or []:
        plt.savefig(str(extra_path), dpi=130)
    plt.close(fig)
    return str(out_path)


def collect_sets():
    sets = []
    for brand_dir in sorted(DATASET_ROOT.iterdir()):
        if not brand_dir.is_dir():
            continue

        golden_path = brand_dir / "golden.jpg"
        if not golden_path.exists():
            continue

        faulty_dirs = [
            d for d in sorted(brand_dir.iterdir())
            if d.is_dir() and d.name.lower().endswith("_faulty")
        ]
        if not faulty_dirs:
            continue

        display_prefix = DISPLAY_NAME_MAP.get(brand_dir.name, brand_dir.name.title())
        index = 1
        for faulty_dir in faulty_dirs:
            candidates = [
                p for p in sorted(faulty_dir.iterdir())
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ]
            for candidate_path in candidates:
                sets.append(
                    {
                        "set_name": f"{display_prefix}_{index}",
                        "golden": golden_path,
                        "candidate": candidate_path,
                        "brand": brand_dir.name,
                        "faulty_folder": faulty_dir.name,
                    }
                )
                index += 1
    return sets


def collect_golden_cases():
    cases = []
    for brand_dir in sorted(DATASET_ROOT.iterdir()):
        if not brand_dir.is_dir():
            continue

        golden_path = brand_dir / "golden.jpg"
        if golden_path.exists():
            cases.append((brand_dir.name, golden_path))
    return cases


def build_faulty_cases():
    return collect_sets()


def build_clean_cases():
    clean_cases = []
    for brand_name, golden_path in collect_golden_cases():
        clean_cases.append(
            {
                "case_type": "golden",
                "brand": brand_name,
                "case_name": f"{slugify(brand_name)}__golden",
                "golden": golden_path,
                "candidate": golden_path,
            }
        )

        golden_bgr = load(golden_path)
        for aug_index, (aug_name, aug_img, aug_path) in enumerate(
            save_augmented_samples(brand_name, golden_path, golden_bgr), start=1
        ):
            clean_cases.append(
                {
                    "case_type": "augmented",
                    "brand": brand_name,
                    "case_name": f"{slugify(brand_name)}__aug_{aug_index:02d}_{aug_name}",
                    "golden": golden_path,
                    "candidate": aug_path,
                    "candidate_image": aug_img,
                }
            )
    return clean_cases


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(records):
    outcome_counts = Counter(record["outcome"] for record in records)
    stage_counts = Counter((record["outcome"], record["stage"]) for record in records)
    expected_counts = Counter(record["expected"] for record in records)

    tp = outcome_counts["TP"]
    tn = outcome_counts["TN"]
    fp = outcome_counts["FP"]
    fn = outcome_counts["FN"]
    total = len(records)
    positive_total = expected_counts["faulty"]
    negative_total = expected_counts["clean"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    lines = []
    lines.append("SUMMARY")
    lines.append(f"  total_samples: {total}")
    lines.append(f"  faulty_samples: {positive_total}")
    lines.append(f"  clean_samples: {negative_total}")
    lines.append(f"  TP: {tp}")
    lines.append(f"  TN: {tn}")
    lines.append(f"  FP: {fp}")
    lines.append(f"  FN: {fn}")
    lines.append(f"  precision: {precision:.4f}")
    lines.append(f"  recall: {recall:.4f}")
    lines.append(f"  specificity: {specificity:.4f}")
    lines.append(f"  npv: {npv:.4f}")
    lines.append(f"  false_positive_rate: {fpr:.4f}")
    lines.append(f"  false_negative_rate: {fnr:.4f}")
    lines.append("")
    lines.append("STAGE BREAKDOWN")
    for outcome in ("TP", "TN", "FP", "FN"):
        for stage in ("Gate 1", "Gate 2", "PASS"):
            count = stage_counts.get((outcome, stage), 0)
            if count:
                lines.append(f"  {outcome} @ {stage}: {count}")
    return "\n".join(lines), {
        "total_samples": total,
        "faulty_samples": positive_total,
        "clean_samples": negative_total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp_gate1": stage_counts.get(("TP", "Gate 1"), 0),
        "tp_gate2": stage_counts.get(("TP", "Gate 2"), 0),
        "fp_gate1": stage_counts.get(("FP", "Gate 1"), 0),
        "fp_gate2": stage_counts.get(("FP", "Gate 2"), 0),
        "tn_pass": stage_counts.get(("TN", "PASS"), 0),
        "fn_pass": stage_counts.get(("FN", "PASS"), 0),
    }


def mistake_score(record):
    if record["outcome"] not in {"FP", "FN"}:
        return 0.0
    return 1.0 - float(record.get("ssim_score") or 0.0)


def build_case_output_dirs(outcome):
    category_dir = ANALYSIS_DIR / outcome
    category_dir.mkdir(parents=True, exist_ok=True)
    return category_dir


def main():
    faulty_cases = build_faulty_cases()
    clean_cases = build_clean_cases()
    all_cases = [
        {**case, "expected": "faulty", "source_kind": "faulty"}
        for case in faulty_cases
    ] + [
        {**case, "expected": "clean", "source_kind": case["case_type"]}
        for case in clean_cases
    ]

    if not all_cases:
        raise RuntimeError(f"No valid golden/faulty sets found under: {DATASET_ROOT}")

    chunks = []
    records = []
    category_examples = defaultdict(int)

    for item in all_cases:
        set_name = item["set_name"] if item.get("source_kind") == "faulty" else item["case_name"]
        golden = load(item["golden"])
        candidate = item.get("candidate_image")
        if candidate is None:
            candidate = load(item["candidate"])

        target_size = estimate_target_size(golden)
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
        report = inspector.inspect(golden, candidate)
        outcome = evaluate_outcome(item["expected"], report)
        stage = stage_label(report)
        case_prefix = f"{outcome}_{stage.replace(' ', '')}_{slugify(set_name)}"
        all_case_path = ALL_REPORTS_DIR / f"{case_prefix}_report.png"

        mirror_paths = []
        if category_examples[outcome] < 2:
            category_dir = build_case_output_dirs(outcome)
            mirror_paths.append(category_dir / f"{case_prefix}_report.png")
            category_examples[outcome] += 1

        out_path = visualize(set_name, golden, candidate, report, all_case_path, mirror_paths=mirror_paths)

        records.append(
            {
                "case_name": set_name,
                "expected": item["expected"],
                "predicted": prediction_from_report(report),
                "outcome": outcome,
                "stage": stage,
                "source_kind": item["source_kind"],
                "brand": item.get("brand", ""),
                "golden": str(item["golden"]),
                "candidate": str(item["candidate"]),
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

        print(f"Saved visualization -> {out_path}\n")

        chunk = [
            "=" * 70,
            set_name,
            "=" * 70,
            report.summary(),
            f"expected: {item['expected']} | predicted: {prediction_from_report(report)} | outcome: {outcome} | stage: {stage}",
            f"source: golden={item['golden']} | candidate={item['candidate']}",
            "",
            f"Saved visualization -> {out_path}",
            "",
        ]
        chunks.append("\n".join(chunk))

    summary_text, summary_counts = summarize_records(records)
    summary_body = "\n\n".join([summary_text] + chunks)

    REPORT_PATH.write_text(summary_body, encoding="utf-8")
    SUMMARY_PATH.write_text(summary_body, encoding="utf-8")

    write_csv(
        PREDICTIONS_PATH,
        [
            "case_name",
            "expected",
            "predicted",
            "outcome",
            "stage",
            "source_kind",
            "brand",
            "golden",
            "candidate",
            "report_path",
            "ssim_score",
            "num_hotspots",
            "gate1_passed",
            "gate2_passed",
        ],
        records,
    )

    write_csv(
        CLASSIFICATION_REPORT_PATH,
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in summary_counts.items()],
    )

    mistakes = [record for record in records if record["outcome"] in {"FP", "FN"}]
    mistakes.sort(key=mistake_score, reverse=True)
    write_csv(
        HARD_MISTAKES_PATH,
        [
            "case_name",
            "expected",
            "predicted",
            "outcome",
            "stage",
            "source_kind",
            "brand",
            "golden",
            "candidate",
            "report_path",
            "ssim_score",
            "num_hotspots",
            "gate1_passed",
            "gate2_passed",
        ],
        mistakes,
    )

    print(summary_text)
    print()
    print(f"Consolidated report saved -> {REPORT_PATH}")
    print(f"Structured summary saved -> {SUMMARY_PATH}")
    print(f"Predictions CSV saved -> {PREDICTIONS_PATH}")
    print(f"Classification summary saved -> {CLASSIFICATION_REPORT_PATH}")
    print(f"Mistakes report saved -> {HARD_MISTAKES_PATH}")


if __name__ == "__main__":
    main()
