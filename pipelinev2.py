"""
Pipeline V2 — Multi-Golden Ensemble Content Gate
=================================================

Why this exists
----------------
`demov2.py` got its low false-positive rate by comparing every candidate
against *synthetically augmented* clean copies of a single golden photo.
Those augmentations (small rotation, brightness jitter, blur/noise,
perspective wobble) are much gentler and more uniform than what real garment
"golden" samples actually look like: different physical units of the same
approved label still carry natural print/fabric tolerance, and they were
photographed by hand under different lighting with a phone camera. Against
a *single* real golden photo, that natural unit-to-unit variation reads as
"anomalous" exactly like a real defect does (see `v3_evaluation`: FP rate
jumped from 25% to 94% once demov3.py switched to real, un-augmented
goldens, while SSIM/severity distributions for FP and TP cases turned out
to overlap almost completely — see analysis in the accompanying report).

Fix implemented here
---------------------
1. **Ensemble matching** (`MultiGoldenContentGate.inspect_ensemble`):
   compare the candidate against *every available real golden crop* for
   that sample (not just one), and keep the best-explained match (lowest
   severity_frac). A clean candidate only needs to resemble *one* of the
   approved physical units closely; a real defect stays anomalous against
   all of them, so recall is preserved while natural golden-to-golden
   variation stops being misread as a fault.

2. **Data-driven threshold calibration**
   (`MultiGoldenContentGate.calibrate_from_goldens`): instead of guessing
   fixed severity/SSIM thresholds, measure the *actual* golden-vs-golden
   variation for this sample (leave-one-out against a reference) and set
   the reject thresholds just above the worst observed natural variation
   (with a safety margin). This makes the gate self-tuning per sample/
   brand instead of relying on one global magic number tuned on augmented
   data that turned out not to represent real variation.

Both changes are opt-in extensions of `pipeline.ContentGate` / `LabelInspector`
— Gate 1 (structural) and Gate 3 (classifier) are reused unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipeline import (
    ContentGate,
    Gate1Result,
    Gate2Result,
    HeuristicHotSpotClassifier,
    InspectionReport,
    StructuralGate,
)


class RobustStructuralGate(StructuralGate):
    """StructuralGate that picks the label contour by aspect-ratio match to
    the expected narrow-ribbon proportions, not just raw contour area.

    Bug found via a v4_evaluation Gate-1 false positive: one candidate photo
    included a wood-grain table background instead of the plain background
    the other crops have. Otsu thresholding picked up the *table texture*
    as the largest connected blob, and its bounding-box skew/size (nothing
    to do with the actual label) got reported as the label's geometry —
    rejecting a perfectly good golden crop for a reason unrelated to the
    label itself. The label ribbon's aspect ratio is known (target_w/h);
    preferring whichever significant contour actually matches it is a much
    stronger signal than "biggest blob" when the frame has real clutter.
    """

    def _select_main_contour(self, significant):
        if len(significant) == 1:
            return significant[0]

        target_ratio = self.target_w / self.target_h

        def aspect_error(c):
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            w, h = min(rw, rh), max(rw, rh)
            if h == 0:
                return float("inf")
            return abs((w / h) - target_ratio)

        by_area = max(significant, key=cv2.contourArea)
        by_aspect = min(significant, key=aspect_error)
        if by_area is by_aspect:
            return by_area

        # Only override the simple "biggest blob" pick when the
        # aspect-based candidate is a clearly, substantially better fit to
        # the known label proportions — otherwise keep the default so this
        # doesn't second-guess the common, uncluttered case.
        if aspect_error(by_aspect) < aspect_error(by_area) * 0.5:
            return by_aspect
        return by_area

    def _skew_is_reliable(self, mask, main_contour):
        h, w = mask.shape[:2]
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        border_fg_frac = float(np.count_nonzero(border)) / border.size
        # These SAM/YOLO crops are frequently cut with little to no
        # background margin, so Otsu thresholding classifies almost the
        # whole frame as "label" and the contour collapses onto the crop
        # boundary itself. minAreaRect's angle on that contour is then
        # dominated by jagged/anti-aliased edge-pixel noise (observed up to
        # ~14 deg of bogus "skew" on visibly straight labels), not a real
        # physical slant. Only trust the skew reading when the contour has
        # genuine background margin around it.
        return border_fg_frac < 0.5


@dataclass
class GoldenCalibration:
    severity_frac_reject_threshold: float
    diff_severity_frac_reject_threshold: float
    max_hotspot_frac_reject_threshold: float
    overall_ssim_reject_threshold: float
    n_pairs: int


class _NoOpPltModule:
    """Swapped in for pipeline.py's module-level `plt` reference during
    ensemble/calibration inspects, so its debug-plot block (plt.figure/
    subplot/imshow/title/axis/tight_layout/close) does nothing instead of
    allocating a real matplotlib figure per call."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return self

        return _noop


_NO_OP_PLT = _NoOpPltModule()


def _subsample(items: Sequence, max_n: int) -> List:
    """Deterministic, evenly-spaced subsample so ensemble/calibration cost
    stays bounded even when a sample folder has 30-40 golden crops."""
    n = len(items)
    if n <= max_n:
        return list(items)
    idx = np.linspace(0, n - 1, max_n).round().astype(int)
    seen = sorted(set(idx.tolist()))
    return [items[i] for i in seen]


def _subsample_paired(items: Sequence, names: Sequence, max_n: int) -> Tuple[List, List]:
    """Same deterministic subsample as `_subsample`, but keeps a parallel
    `names` list (e.g. golden filenames) in sync with the chosen items —
    used so diagnostics/plots can report which actual golden was matched
    against, not just an anonymous array."""
    n = len(items)
    if n <= max_n:
        return list(items), list(names)
    idx = np.linspace(0, n - 1, max_n).round().astype(int)
    seen = sorted(set(idx.tolist()))
    return [items[i] for i in seen], [names[i] for i in seen]


class MultiGoldenContentGate(ContentGate):
    """ContentGate extended to reason over an ensemble of real golden
    samples instead of a single fixed reference photo."""

    # Crops in this dataset are cut tight to the label edges, so a few
    # pixels of crop-boundary inconsistency between golden and candidate
    # (different tightness of crop, warp interpolation at the frame edge)
    # land right at the image border. Real print defects (ink bleed, missing
    # stitch, lamination scuff, text mismatch) sit inside the label body, not
    # in a thin rim hugging the outer crop boundary — so any hotspot that
    # touches the border within this fraction of the shorter image
    # dimension is treated as a crop/registration artifact, not a defect.
    DEFAULT_BORDER_MARGIN_FRAC = 0.04

    # cv2.findTransformECC returns its own correlation coefficient (0-1) for
    # the warp it converged to. Accepting *any* converged warp regardless of
    # this score was a real bug found via diagnosis: on a mostly-uniform,
    # sparsely-printed label, ECC can converge to a low-quality alignment
    # (cc as low as 0.62 observed) that nonetheless looks "sane" by the area
    # check, and which silently replaced pipeline.py's correct behavior for
    # a genuine defect (reject due to registration failure) with a bogus
    # "clean-looking" match that hid the defect entirely (severity_frac
    # dropped to 0.0 for a case that should have been an obvious reject).
    MIN_ECC_CORRELATION = 0.75

    # A blanket "SSIM >= 0.90 overrides any reject" safety net was tried and
    # measured to be net-harmful: recomputing pre-override verdicts across
    # the FP/FN mistakes in a full v4_evaluation run showed the FP cases it
    # was meant to rescue mostly sit *below* 0.90 SSIM already (so the rule
    # barely helps them), while genuine defects that trip diff_severity_frac
    # by a modest margin very often *do* clear 0.90 SSIM (a small but real
    # localized flaw doesn't move whole-image SSIM much) and were being
    # silently flipped from correct rejects to false negatives. There's no
    # clean separation between "benign golden variation" and "real small
    # defect" using SSIM + these metrics alone, so the rule was removed
    # rather than tuned further.

    def __init__(
        self,
        *args,
        max_ensemble_size: int = 6,
        debug_plots: bool = False,
        border_margin_frac: float = DEFAULT_BORDER_MARGIN_FRAC,
        detect_rotation: bool = True,
        min_ecc_correlation: float = MIN_ECC_CORRELATION,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_ensemble_size = max_ensemble_size
        self.debug_plots = debug_plots
        self.border_margin_frac = border_margin_frac
        self.detect_rotation = detect_rotation
        self.min_ecc_correlation = min_ecc_correlation

    # `pipeline.ContentGate.inspect` unconditionally builds a matplotlib
    # debug figure on every call, which is pure overhead once you're calling
    # it dozens of times per candidate for ensemble matching/calibration.
    # Swap out pipeline.py's module-level `plt` reference for a no-op stub
    # for the duration of the call so no figure is ever created (rather than
    # letting figures get created and only skipping show/close, which leaks
    # memory once pipeline.py's own plt.close(fig) call becomes a no-op too).
    def inspect(self, golden_bgr, candidate_bgr) -> Gate2Result:
        # Orientation correction is done per (golden, candidate) PAIR, not
        # once for the whole ensemble against an arbitrary reference. Bug
        # found via a v4_evaluation report: rotation detection used to run
        # once against ensemble[0]; if that particular golden happened to
        # share the candidate's orientation, k=0 was "correctly" decided
        # against it — but ensemble matching then picked a *different*
        # golden with a different orientation, and the still-unrotated
        # candidate got compared against it, falling through to a plain
        # resize that visibly stretches/shears the content. Sample folders
        # can genuinely mix orientations (confirmed: sample1's golden
        # folder has 20 portrait + 9 landscape crops), so every pair needs
        # its own decision.
        rotation_k = 0
        if self.detect_rotation:
            rotation_k = self._detect_orientation(golden_bgr, candidate_bgr)
            if rotation_k != 0:
                candidate_bgr = self._rotate_bgr(candidate_bgr, rotation_k)

        raw = self._inspect_raw(golden_bgr, candidate_bgr)
        filtered = self._exclude_border_hotspots(raw)

        debug_info = dict(getattr(self, "_last_register_debug", {}))
        debug_info["n_hotspots_before_border_filter"] = len(raw.hotspots)
        debug_info["n_hotspots_after_border_filter"] = len(filtered.hotspots)
        debug_info["n_border_excluded"] = len(raw.hotspots) - len(filtered.hotspots)
        debug_info["coverage_frac"] = self._warp_coverage_frac(filtered, candidate_bgr)
        debug_info["rotation_k"] = rotation_k
        filtered.debug_info = debug_info
        return filtered

    @staticmethod
    def _warp_coverage_frac(result: Gate2Result, candidate_bgr: np.ndarray) -> float:
        """Fraction of the golden-sized frame that's actually covered by
        real candidate content after warping (vs. left as warpPerspective's
        black out-of-bounds padding). 1.0 for the plain-resize fallback
        (no H at all — the whole frame is trivially filled), lower values
        flag a warp that only managed to align a small sliver of the frame,
        which is a strong signal that the comparison downstream is
        unreliable regardless of what severity_frac says."""
        if result.homography is None or result.aligned_candidate is None:
            return 1.0
        h, w = result.aligned_candidate.shape[:2]
        cand_h, cand_w = candidate_bgr.shape[:2]
        ones = np.full((cand_h, cand_w), 255, dtype=np.uint8)
        valid = cv2.warpPerspective(ones, result.homography, (w, h))
        return float(np.count_nonzero(valid)) / float(h * w)

    def _inspect_raw(self, golden_bgr, candidate_bgr) -> Gate2Result:
        if self.debug_plots:
            return super().inspect(golden_bgr, candidate_bgr)
        import pipeline as _pipeline_module

        orig_plt = _pipeline_module.plt
        _pipeline_module.plt = _NO_OP_PLT
        try:
            return super().inspect(golden_bgr, candidate_bgr)
        finally:
            _pipeline_module.plt = orig_plt

    def _exclude_border_hotspots(self, result: Gate2Result) -> Gate2Result:
        """Drop hotspots that touch the crop boundary (crop-alignment
        artifacts, not defects), then recompute the severity/diff/max-area
        fractions and the pass/fail decision from the surviving hotspots
        only. Mirrors the tail of `pipeline.ContentGate.inspect` exactly,
        just re-run on the filtered hotspot set."""
        if not result.hotspots or result.ssim_map is None:
            return result

        h, w = result.ssim_map.shape[:2]
        margin = max(1, int(round(self.border_margin_frac * min(h, w))))

        def border_overlap_frac(bbox):
            # Fraction of the hotspot's own area that falls within the
            # border band. Merely *touching* an edge isn't enough to
            # exclude — a genuine defect spanning most/all of the crop
            # necessarily touches every edge too. Only a hotspot whose
            # footprint is mostly *confined* to the border band (a thin rim
            # artifact hugging the crop boundary) should be dropped.
            x, y, ww, hh = bbox
            total_area = ww * hh
            if total_area == 0:
                return 0.0
            ix0, iy0 = max(x, margin), max(y, margin)
            ix1, iy1 = min(x + ww, w - margin), min(y + hh, h - margin)
            interior_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            return (total_area - interior_area) / total_area

        BORDER_OVERLAP_EXCLUDE_THRESHOLD = 0.6
        kept = [
            hs for hs in result.hotspots
            if border_overlap_frac(hs.bbox) < BORDER_OVERLAP_EXCLUDE_THRESHOLD
        ]
        n_excluded = len(result.hotspots) - len(kept)
        if n_excluded == 0:
            return result

        old_hotspot_area = sum(hs.area for hs in result.hotspots)
        fg_area = (old_hotspot_area / result.hotspot_area_frac) if result.hotspot_area_frac else None

        weighted_severity = 0.0
        diff_weighted_severity = 0.0
        max_hotspot_area = 0
        hotspot_area = 0
        for hs in kept:
            x, y, ww, hh = hs.bbox
            local_ssim = result.ssim_map[y:y + hh, x:x + ww]
            local_diff = result.diff_map[y:y + hh, x:x + ww]
            severity_weight = max(0.0, 1.0 - float(local_ssim.mean()))
            diff_weight = float(local_diff.mean()) / 255.0
            weighted_severity += hs.area * severity_weight
            diff_weighted_severity += hs.area * diff_weight
            max_hotspot_area = max(max_hotspot_area, hs.area)
            hotspot_area += hs.area

        if fg_area:
            severity_frac = weighted_severity / fg_area
            diff_severity_frac = diff_weighted_severity / fg_area
            max_hotspot_frac = max_hotspot_area / fg_area
            hotspot_area_frac = hotspot_area / fg_area
            reject_hotspot_area = max(self.min_hotspot_area * 3, int(0.01 * fg_area))
        else:
            severity_frac = diff_severity_frac = max_hotspot_frac = hotspot_area_frac = 0.0
            reject_hotspot_area = self.min_hotspot_area * 3

        high_ssim_hotspot_count = 10
        reasons: List[str] = []
        if result.ssim_score < self.overall_ssim_reject_threshold:
            if hotspot_area >= reject_hotspot_area:
                reasons.append(
                    f"Overall SSIM {result.ssim_score:.3f} below threshold "
                    f"{self.overall_ssim_reject_threshold} with {hotspot_area}px of hotspot evidence "
                    f"(after excluding {n_excluded} border-adjacent hotspot(s))."
                )
        elif len(kept) >= high_ssim_hotspot_count:
            reasons.append(
                f"High-SSIM sample still has {len(kept)} hot spots after border exclusion — "
                f"likely a real defect cluster."
            )

        severity_reject = severity_frac >= self.severity_frac_reject_threshold
        diff_severity_reject = diff_severity_frac >= self.diff_severity_frac_reject_threshold
        max_hotspot_reject = max_hotspot_frac >= self.max_hotspot_frac_reject_threshold
        if severity_reject or max_hotspot_reject:
            reasons.append(
                f"Localized textured defect evidence (border-excluded): severity_frac={severity_frac:.3f} "
                f"(threshold {self.severity_frac_reject_threshold}), max_hotspot_frac={max_hotspot_frac:.3f} "
                f"(threshold {self.max_hotspot_frac_reject_threshold})."
            )
        if diff_severity_reject:
            reasons.append(
                f"Localized flat/tonal defect evidence (border-excluded): "
                f"diff_severity_frac={diff_severity_frac:.3f} (threshold {self.diff_severity_frac_reject_threshold})."
            )
        if kept:
            reasons.append(
                f"{len(kept)} hot spot(s) require classification "
                f"({n_excluded} border-adjacent hotspot(s) excluded)."
            )

        passed = not (
            (result.ssim_score < self.overall_ssim_reject_threshold and hotspot_area >= reject_hotspot_area)
            or (len(kept) >= high_ssim_hotspot_count)
            or severity_reject
            or max_hotspot_reject
            or diff_severity_reject
        )

        return dataclasses.replace(
            result,
            hotspots=kept,
            hotspot_area_frac=float(hotspot_area_frac),
            max_hotspot_area_frac=float(max_hotspot_frac),
            severity_frac=float(severity_frac),
            diff_severity_frac=float(diff_severity_frac),
            passed=passed,
            reasons=reasons,
        )

    # These label crops are tight, low-background-texture regions, so ORB
    # frequently can't find the >= min_good_matches it needs (observed in
    # ~70/258 v4 cases: "registration matches used: 0"). pipeline.py's
    # fallback for that case is a plain resize, which leaves real
    # scale/rotation/translation offsets between candidate and golden
    # uncorrected — those misaligned edges then get scored as large false
    # "hot spots", which was the dominant source of the remaining false
    # positives even after ensemble matching + calibration. ECC (enhanced
    # correlation coefficient) alignment works directly off pixel
    # intensities rather than sparse keypoints, so it can still recover a
    # decent Euclidean (rotation+translation) alignment on these low-texture
    # crops where ORB comes up empty.
    def register(self, golden_gray: np.ndarray, candidate_gray: np.ndarray):
        # Diagnostic breadcrumb for DEBUG tooling: which registration path
        # actually produced the alignment used downstream. Overwritten on
        # every call, read by `inspect()` immediately after.
        self._last_register_debug = {
            "registration_method": "unknown",
            "num_good_matches": 0,
            "ecc_correlation": None,
        }

        H, n_matches = super().register(golden_gray, candidate_gray)
        self._last_register_debug["num_good_matches"] = n_matches
        if H is not None:
            self._last_register_debug["registration_method"] = "orb"
            return H, n_matches

        h, w = golden_gray.shape[:2]
        cand_h, cand_w = candidate_gray.shape[:2]
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
        try:
            golden_f = golden_gray.astype(np.float32)
            # ECC needs both images the same size; estimate the warp in
            # golden-sized coordinates, then fold the candidate->golden
            # resize scale into the final matrix so it applies correctly to
            # the full-resolution candidate_bgr that pipeline.py warps.
            cand_f = cv2.resize(candidate_gray, (w, h)).astype(np.float32)
            cc, warp = cv2.findTransformECC(
                golden_f, cand_f, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
            )
        except cv2.error:
            self._last_register_debug["registration_method"] = "resize_fallback"
            self._last_register_debug["ecc_error"] = True
            return None, n_matches

        self._last_register_debug["ecc_correlation"] = float(cc)
        if cc < self.min_ecc_correlation:
            # Poor-quality convergence — trust pipeline.py's own registration
            # -failure handling (reject / plain-resize fallback) rather than
            # an alignment ECC itself isn't confident about.
            self._last_register_debug["registration_method"] = "resize_fallback"
            self._last_register_debug["ecc_rejected_low_confidence"] = True
            return None, n_matches

        scale_x, scale_y = w / cand_w, h / cand_h
        A = warp[:, :2] @ np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=np.float32)
        t = warp[:, 2:3]
        H_ecc = np.vstack([np.hstack([A, t]), [0.0, 0.0, 1.0]]).astype(np.float64)
        if not self._is_sane_homography(H_ecc, cand_w, cand_h):
            self._last_register_debug["registration_method"] = "resize_fallback"
            self._last_register_debug["ecc_rejected_insane_homography"] = True
            return None, n_matches
        self._last_register_debug["registration_method"] = "ecc"
        return H_ecc, n_matches

    # cv2 rotation flags for a coarse 0/90/180/270 orientation search, keyed
    # by how many 90-degree clockwise turns they represent.
    _ROTATE_FLAGS = {
        0: None,
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    @classmethod
    def _rotate_bgr(cls, img: np.ndarray, k: int) -> np.ndarray:
        flag = cls._ROTATE_FLAGS[k % 4]
        return img if flag is None else cv2.rotate(img, flag)

    def _detect_orientation(self, reference_bgr: np.ndarray, candidate_bgr: np.ndarray) -> int:
        """Coarse 0/90/180/270 orientation search: ORB registration (and its
        ECC fallback) is only good for small angles — a candidate captured
        genuinely sideways gives 0 ORB matches and ECC won't converge either
        (it's a local optimizer seeded from identity, nowhere near a 90 deg
        offset). Try each canonical rotation of the candidate against a
        reference golden and keep the one with the most ORB matches, so the
        real (small-angle) registration step downstream starts from
        approximately the right orientation instead of failing outright."""
        reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        match_counts = {}
        for k in (0, 1, 2, 3):
            candidate_gray = cv2.cvtColor(self._rotate_bgr(candidate_bgr, k), cv2.COLOR_BGR2GRAY)
            # Base ORB-only registration here (not the ECC-augmented
            # override) — cheap, and ECC's convergence isn't a meaningful
            # signal for "which orientation is right" the way match count is.
            _, n_matches = ContentGate.register(self, reference_gray, candidate_gray)
            match_counts[k] = n_matches

        best_k = max(match_counts, key=match_counts.get)
        baseline = max(match_counts[0], 1)
        # Only trust the rotated orientation if it's a clear, well-matched
        # winner — otherwise a low-texture crop with near-zero matches in
        # every orientation could flip an already-correct image on noise.
        if best_k != 0 and match_counts[best_k] >= self.min_good_matches and match_counts[best_k] >= 1.5 * baseline:
            return best_k

        # ORB found (near-)zero matches in every orientation — inconclusive,
        # not evidence of "no rotation needed". This is common on these
        # crops (sparse keypoints on mostly-uniform fabric); confirmed via
        # DEBUG dump that a genuine 90-deg orientation mismatch inside
        # sample1 (some golden crops stored portrait, some landscape) was
        # silently missed this way. Fall back to comparing ECC correlation
        # (pixel-intensity based, doesn't need keypoints) across the 4
        # orientations instead.
        if max(match_counts.values()) < self.min_good_matches:
            return self._detect_orientation_ecc(reference_bgr, candidate_bgr)
        return 0

    def _detect_orientation_ecc(self, reference_bgr: np.ndarray, candidate_bgr: np.ndarray) -> int:
        """Coarse translation-only ECC correlation at each of the 4
        orientations — cheap (MOTION_TRANSLATION converges fast) and works
        directly off pixel intensity, so it doesn't need the keypoints ORB
        couldn't find. Used only when ORB's match-count signal was
        inconclusive in every orientation."""
        h, w = reference_bgr.shape[:2]
        reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

        correlations = {}
        for k in (0, 1, 2, 3):
            rotated_gray = cv2.cvtColor(self._rotate_bgr(candidate_bgr, k), cv2.COLOR_BGR2GRAY)
            candidate_f = cv2.resize(rotated_gray, (w, h)).astype(np.float32)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cc, _ = cv2.findTransformECC(
                    reference_gray, candidate_f, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
                )
            except cv2.error:
                cc = -1.0
            correlations[k] = cc

        best_k = max(correlations, key=correlations.get)
        # Require both a genuinely good fit AND a clear margin over the
        # as-captured orientation — a mediocre "best" is more likely noise
        # than a real rotation.
        if (
            best_k != 0
            and correlations[best_k] >= 0.5
            and correlations[best_k] >= correlations[0] + 0.1
        ):
            return best_k

        # ECC also inconclusive (common on very low-texture crops where even
        # a translation-only fit won't converge cleanly). Last resort:
        # aspect ratio. Found via a v4_evaluation case where a 859x245
        # (landscape) candidate was compared against a 317x990 (portrait)
        # golden with 0 ORB matches in every orientation and ECC maxing out
        # at 0.32 correlation (below the 0.5 bar above) — silently defaulted
        # to "no rotation" and produced a visibly smeared/sheared alignment.
        # Confirmed the same golden folders mix portrait and landscape crops
        # in multiple samples, so this isn't a one-off.
        return self._detect_orientation_aspect(reference_bgr, candidate_bgr)

    def _detect_orientation_aspect(self, reference_bgr: np.ndarray, candidate_bgr: np.ndarray) -> int:
        """Last-resort orientation signal when both ORB and ECC are
        inconclusive: the candidate's own aspect ratio vs. the golden's.
        These garment ribbon crops are always much taller than wide (or vice
        versa) — a landscape candidate against a portrait golden (or vice
        versa) is essentially never the correct 0-degree orientation, so
        this is strong evidence even without any texture/keypoints."""
        ref_h, ref_w = reference_bgr.shape[:2]
        ref_ratio = ref_w / ref_h
        cand_h, cand_w = candidate_bgr.shape[:2]

        def rotated_ratio(k):
            w, h = (cand_w, cand_h) if k % 2 == 0 else (cand_h, cand_w)
            return w / h

        errors = {k: abs(np.log(rotated_ratio(k) / ref_ratio)) for k in (0, 1, 2, 3)}
        best_k = min(errors, key=errors.get)
        # Only override "no rotation" when the as-captured aspect ratio is a
        # clearly bad fit and the rotated one is a clearly better one —
        # avoids acting on noise for near-square crops where aspect ratio
        # alone can't distinguish portrait from landscape.
        if best_k != 0 and errors[0] > np.log(1.3) and errors[best_k] < errors[0] * 0.4:
            return best_k
        return 0

    def inspect_ensemble(
        self,
        golden_bgr_list: Sequence[np.ndarray],
        candidate_bgr: np.ndarray,
        return_best_golden: bool = False,
        golden_names: Optional[Sequence[str]] = None,
    ):
        """Score the candidate against every golden in the ensemble and keep
        the MEDIAN-ranked match — i.e. "does this candidate look like a
        clean copy of *most* approved physical units?" rather than "does it
        look like this one specific photo?".

        Earlier this picked the single best (lowest-severity) match instead
        of the median. That let a genuine defect hide behind one
        accidentally-lucky pairing: confirmed via a real fresh_v4_eval FN
        where 2 of 3 subsampled goldens correctly registered the defect
        (severity 0.279, 0.214, both well past the reject threshold) but the
        3rd happened to register in a way that dampened the visible
        difference (severity 0.081) — min() picked exactly that one and the
        candidate wrongly passed. A clean candidate should resemble most of
        the ensemble, not just the single most forgiving comparison; the
        median is far more robust to one geometrically-lucky/unlucky pairing
        while still absorbing genuine unit-to-unit golden variation.

        `golden_names`, if given, must be parallel to `golden_bgr_list` (e.g.
        source filenames) — when `return_best_golden` is also set, the name
        of the actually-matched golden is returned too, so callers/plots can
        report *which* golden a verdict was made against instead of an
        anonymous array.
        """
        if golden_names is not None:
            ensemble, ensemble_names = _subsample_paired(
                list(golden_bgr_list), list(golden_names), self.max_ensemble_size
            )
        else:
            ensemble = _subsample(list(golden_bgr_list), self.max_ensemble_size)
            ensemble_names = None

        # Orientation is now decided per (golden, candidate) pair inside
        # `inspect()` itself — see the comment there for why a single
        # ensemble-wide decision was wrong.
        results = [self.inspect(g, candidate_bgr) for g in ensemble]
        order = sorted(
            range(len(results)),
            key=lambda i: (results[i].severity_frac, results[i].diff_severity_frac, -results[i].ssim_score),
        )
        median_idx = order[len(order) // 2]
        if return_best_golden:
            matched_name = ensemble_names[median_idx] if ensemble_names is not None else None
            return results[median_idx], ensemble[median_idx], matched_name
        return results[median_idx]

    # How far calibration is allowed to loosen the class defaults. With only
    # ~9 golden pairs to estimate from, a single badly-registered pair (e.g.
    # a golden crop with heavy motion blur) can otherwise blow a percentile
    # up to a nonsensical value (>1.0 for an area fraction) and silently
    # disable that reject check entirely, which is what tanked recall to
    # ~39% in the first version of this calibration (see v4_evaluation run
    # log: max_hotspot thresholds calibrated as high as 1.58, i.e. never
    # triggerable). Capping the loosening keeps calibration a *tuning*
    # mechanism, not a way to accidentally turn a gate off.
    #
    # Raised 1.8 -> 3.5 once calibration itself switched to measuring the
    # median-of-ensemble statistic (see calibrate_from_goldens): the
    # original worry (one bad pair blowing up the estimate) is now much
    # less applicable, since each calibration sample is itself already a
    # median-of-5 pick, robust to any single bad pairing. Measured directly
    # on sample1 (embossed, lighting-sensitive label — the highest natural-
    # variation brand in this dataset): the robust median-of-ensemble
    # diff_severity_frac needed a threshold of ~0.147 against a class
    # default of 0.05 (2.9x) — the old 1.8x cap silently left genuinely
    # clean goldens rejected no matter how they were calibrated.
    MAX_LOOSEN_FACTOR = 3.5
    MIN_SSIM_FLOOR = 0.55

    @staticmethod
    def _robust_upper_bound(values: Sequence[float], k: float = 3.0) -> float:
        """median + k * MAD (scaled to be std-consistent), far less sensitive
        to a single outlier pair than a percentile computed from ~9 samples."""
        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) * 1.4826
        return median + k * mad

    def calibrate_from_goldens(
        self,
        golden_bgr_list: Sequence[np.ndarray],
        min_pairs: int = 3,
        max_pairs: int = 10,
    ) -> Optional[GoldenCalibration]:
        """Leave-one-out against a reference golden: measure how far real
        golden units naturally drift from one another, and set reject
        thresholds just above that natural variation (median + 3*MAD),
        capped so a calibration run can only loosen thresholds up to
        `MAX_LOOSEN_FACTOR` times the class defaults. Returns None (caller
        keeps the class defaults) when there aren't enough golden samples to
        trust the estimate.

        Measures the SAME statistic `inspect_ensemble` actually decides on
        (the median-of-ensemble pick), not a single reference-vs-other pair.
        These are not interchangeable: the median needs a candidate to
        resemble most of the ensemble, not just one lucky member, so it's
        typically higher (stricter) than any single random pair. Calibrating
        against single pairs under-estimated natural variation and rejected
        genuinely clean goldens once decisions switched to the median-based
        pick (confirmed: FP jumped to 89/182 with single-pair calibration
        before this fix)."""
        goldens = _subsample(list(golden_bgr_list), max_pairs + 1)
        if len(goldens) < min_pairs + 1:
            return None

        severities, diff_severities, max_fracs, ssims = [], [], [], []
        for i, held_out in enumerate(goldens):
            rest = goldens[:i] + goldens[i + 1:]
            r = self.inspect_ensemble(rest, held_out)
            severities.append(r.severity_frac)
            diff_severities.append(r.diff_severity_frac)
            max_fracs.append(r.max_hotspot_area_frac)
            ssims.append(r.ssim_score)

        def capped(default, values):
            return min(
                max(default, self._robust_upper_bound(values)),
                default * self.MAX_LOOSEN_FACTOR,
            )

        ssim_floor_estimate = float(np.median(ssims)) - 3.0 * float(
            np.median(np.abs(np.asarray(ssims) - np.median(ssims)))
        ) * 1.4826
        calibrated_ssim = min(self.overall_ssim_reject_threshold, ssim_floor_estimate)
        calibrated_ssim = max(calibrated_ssim, self.MIN_SSIM_FLOOR)

        return GoldenCalibration(
            severity_frac_reject_threshold=capped(self.severity_frac_reject_threshold, severities),
            diff_severity_frac_reject_threshold=capped(self.diff_severity_frac_reject_threshold, diff_severities),
            max_hotspot_frac_reject_threshold=capped(self.max_hotspot_frac_reject_threshold, max_fracs),
            overall_ssim_reject_threshold=calibrated_ssim,
            n_pairs=len(severities),
        )

    def apply_calibration(self, cal: Optional[GoldenCalibration]) -> None:
        if cal is None:
            return
        self.severity_frac_reject_threshold = cal.severity_frac_reject_threshold
        self.diff_severity_frac_reject_threshold = cal.diff_severity_frac_reject_threshold
        self.max_hotspot_frac_reject_threshold = cal.max_hotspot_frac_reject_threshold
        self.overall_ssim_reject_threshold = cal.overall_ssim_reject_threshold


class LabelInspectorV2:
    """Gate 1 (unchanged) -> Gate 2 (multi-golden ensemble) -> Gate 3
    (unchanged). Mirrors `pipeline.LabelInspector`'s orchestration but takes
    a *list* of golden images for Gate 2 instead of a single one, and can
    self-calibrate its Gate 2 thresholds from that same list."""

    def __init__(
        self,
        target_size_px: Tuple[float, float],
        gate1_kwargs: Optional[dict] = None,
        gate2_kwargs: Optional[dict] = None,
        classifier=None,
        max_ensemble_size: int = 6,
    ):
        self.gate1 = RobustStructuralGate(target_size_px, **(gate1_kwargs or {}))
        self.gate2 = MultiGoldenContentGate(max_ensemble_size=max_ensemble_size, **(gate2_kwargs or {}))
        self.classifier = classifier or HeuristicHotSpotClassifier()

    def calibrate(self, golden_bgr_list: Sequence[np.ndarray], **kwargs) -> Optional[GoldenCalibration]:
        cal = self.gate2.calibrate_from_goldens(golden_bgr_list, **kwargs)
        self.gate2.apply_calibration(cal)
        return cal

    def inspect(
        self,
        golden_bgr_list: Sequence[np.ndarray],
        candidate_bgr: np.ndarray,
        golden_names: Optional[Sequence[str]] = None,
    ) -> InspectionReport:
        g1 = self.gate1.inspect(candidate_bgr)
        if not g1.passed:
            return InspectionReport(verdict="REJECT_GATE1", gate1=g1)

        g2, matched_golden_bgr, matched_golden_name = self.gate2.inspect_ensemble(
            golden_bgr_list, candidate_bgr, return_best_golden=True, golden_names=golden_names
        )
        for hs in g2.hotspots:
            cls, conf = self.classifier.classify(hs)
            hs.defect_class = cls
            hs.confidence = conf

        verdict = "PASS" if g2.passed else "REJECT_GATE2"
        report = InspectionReport(verdict=verdict, gate1=g1, gate2=g2)
        # Dynamic attributes (InspectionReport isn't frozen): the golden that
        # was *actually* matched/aligned against (image + its source name),
        # for diagnostics/plots that want to show the real comparison
        # instead of an arbitrary ensemble member — see the demov4.py
        # visualization bug this was added to fix.
        report.matched_golden_bgr = matched_golden_bgr
        report.matched_golden_name = matched_golden_name
        return report
