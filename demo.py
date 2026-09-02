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

import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import LabelInspector, StructuralGate
import demov3

UPLOAD_DIR = "label_dataset/label_dataset"
OUT_DIR = "./brute"
os.makedirs(OUT_DIR, exist_ok=True)

SETS = {
    "Zara_1": {
        "golden": r"F:\label_defect\single_golden_test_input\zara\golden\golden.jpg",
        "candidate": r"F:\label_defect\single_golden_test_input\zara\faulty\CamScanner 07-04-2026 12.23_5.jpg",
    },
}


def load(path, max_dim=700):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# def estimate_target_size(golden_bgr):
#     """Derive a Gate-1 target size directly from the golden sample's own
#     minAreaRect, since we only have one reference photo per set (no fixed
#     calibrated rig here). In production, replace with a constant measured
#     from the approved physical spec."""
#     tmp_gate = StructuralGate(target_size_px=(1, 1))  # dummy target, unused here
#     gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     mask = tmp_gate._binarize(gray)
#     # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # main = max(contours, key=cv2.contourArea)
#     # rect = cv2.minAreaRect(main)
#     ys, xs = np.where(mask > 0)

#     pts = np.column_stack((xs, ys)).astype(np.float32)

#     rect = cv2.minAreaRect(pts)
#     w, h = rect[1]
#     print(w,h)
#     mask = tmp_gate._binarize(gray)


#     return (min(w, h), max(w, h))
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

def visualize(set_name, golden_bgr, candidate_bgr, report):
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
    out_path = os.path.join(OUT_DIR, f"{set_name}_report.png")
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    for set_name, paths in SETS.items():
        golden_path = Path(paths["golden"])
        golden = load(str(golden_path))
        candidate = load(paths["candidate"])

        target_size = estimate_target_size(golden)
        print(f"target:{target_size}")

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

        # -- Per-sample adaptive severity threshold (same mechanism as
        # demov3.py) -- a single global severity_frac/diff_severity_frac
        # threshold is tuned to sit above the noisiest sample's own clean-
        # photo baseline, which silently misses real defects on cleaner
        # labels (this is exactly what caused this script's earlier FN).
        # With only one golden here, there's no real leave-one-out data to
        # probe, so calibrate from synthetic photographic-noise variants
        # of that one golden instead (in-memory only, never added to the
        # actual reference used for inspection).
        synthetic_imgs = demov3.synthesize_probe_goldens([golden_path], demov3.SYNTHETIC_PROBE_COUNT, set_name)
        severity_baseline, diff_baseline = [], []
        for synth in synthetic_imgs:
            probe_report, _ = demov3.best_match_report(inspector, [golden], synth)
            g2 = probe_report.gate2
            if g2 is not None:
                if g2.severity_frac is not None:
                    severity_baseline.append(g2.severity_frac)
                if g2.diff_severity_frac is not None:
                    diff_baseline.append(g2.diff_severity_frac)

        global_sev = inspector.gate2.severity_frac_reject_threshold
        global_diff = inspector.gate2.diff_severity_frac_reject_threshold
        adaptive_sev = demov3._adaptive_threshold(severity_baseline, global_sev)
        adaptive_diff = demov3._adaptive_threshold(diff_baseline, global_diff)
        inspector.gate2.severity_frac_reject_threshold = adaptive_sev
        inspector.gate2.diff_severity_frac_reject_threshold = adaptive_diff
        print(
            f"adaptive thresholds ({len(synthetic_imgs)} synthetic probes): "
            f"severity_frac {global_sev:.4f} -> {adaptive_sev:.4f}; "
            f"diff_severity_frac {global_diff:.4f} -> {adaptive_diff:.4f}"
        )

        report = inspector.inspect(golden, candidate)

        print("=" * 70)
        print(set_name)
        print("=" * 70)
        print(report.summary())
        print()

        out_path = visualize(set_name, golden, candidate, report)
        print(report.gate2.reasons)
        print(f"Saved visualization -> {out_path}\n")


if __name__ == "__main__":
    main()
