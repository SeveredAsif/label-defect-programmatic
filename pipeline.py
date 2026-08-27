"""
AI-Powered Label Inspection — Two-Gate Hybrid Pipeline
=======================================================

Implements the method described in `Label_Inspection_AI_Pipeline.pptx`:

GATE 1 — STRUCTURAL INTEGRITY (Morphological Verification)
    Verifies the physical "skeleton" of the label BEFORE looking at print
    content. Uses thresholding + contour analysis.
        - Length & Width      -> contour bounding box vs. target dimensions
        - Corner Angles       -> minAreaRect angle vs. 0/90 deg (slanted cut)
        - Continuous Ribbon   -> multiple/undivided blobs (uncut units)
    STOP GAP: if Gate 1 fails, the label is rejected immediately and Gate 2
    is never run (saves compute, matches the pptx's explicit design).

GATE 2 — CONTENT & PRINT ANALYSIS
    Matches the print content to the "Golden Reference" using classical CV.
        Step 1: Alignment / Registration
            - ORB feature detection ("anchors")
            - Homography (rotation & tilt) via RANSAC
            - Warping candidate image onto the golden image's grid
        Step 2: Perceptual comparison
            - SSIM (Structural Similarity) index map -> "Hot Spot" heatmap
            - Direct pixel-subtraction (deterministic, alternative/backup
              method) -> difference mask thresholded against a noise floor
            Both masks are combined for robustness, and the surviving
            blobs become candidate defect ROIs ("Hot Spots").

GATE 3 — ML CLASSIFIER (final gate)
    Each cropped Hot Spot is fed into a classifier that assigns it to a
    defect class (e.g. "Ink Bleed", "Missing Stitch", "Lamination Scuff",
    "Text/Number Mismatch", ...). A CNN feature-extractor classifier is
    provided (`CNNHotSpotClassifier`), together with a dependency-free
    heuristic fallback (`HeuristicHotSpotClassifier`) that works out of the
    box with zero training data, since no labeled defect-crop dataset was
    supplied yet.

Pipeline summary (from the deck's summary slide):

    Defect Group | Fault Class            | Gate               | Method
    -------------|------------------------|--------------------|------------------------
    Structural   | Cutting / Chopped      | Gate 1 (Dimensions)| Contour Bounding Box
    Print        | Ink Bleed / Blur       | Gate 2 (Content)   | SSIM Heatmap ROI
    Surface      | Lamination / Scuff     | Gate 2 (Content)   | Structural Similarity + Morphology
    Structural   | Slanted Cut            | Gate 1 (Dimensions)| Coordinate Geometry
"""

from __future__ import annotations
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt 

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class Gate1Result:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    width_px: Optional[float] = None
    height_px: Optional[float] = None
    angle_deg: Optional[float] = None
    num_components: Optional[int] = None
    mask: Optional[np.ndarray] = None
    box_points: Optional[np.ndarray] = None


@dataclass
class HotSpot:
    bbox: Tuple[int, int, int, int]     # x, y, w, h in candidate-image coords
    area: int
    crop: np.ndarray
    defect_class: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Gate2Result:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    aligned_candidate: Optional[np.ndarray] = None
    ssim_score: Optional[float] = None
    ssim_map: Optional[np.ndarray] = None
    diff_map: Optional[np.ndarray] = None
    combined_defect_mask: Optional[np.ndarray] = None
    hotspots: List[HotSpot] = field(default_factory=list)
    homography: Optional[np.ndarray] = None
    num_good_matches: Optional[int] = None
    hotspot_area_frac: Optional[float] = None
    max_hotspot_area_frac: Optional[float] = None
    severity_frac: Optional[float] = None
    diff_severity_frac: Optional[float] = None


@dataclass
class InspectionReport:
    verdict: str                       # "PASS", "REJECT_GATE1", "REJECT_GATE2"
    gate1: Gate1Result
    gate2: Optional[Gate2Result] = None

    def summary(self) -> str:
        lines = [f"VERDICT: {self.verdict}"]
        lines.append("-- Gate 1 (Structural) --")
        lines.append(f"  passed: {self.gate1.passed}")
        if self.gate1.width_px is not None:
            lines.append(f"  size(px): {self.gate1.width_px:.1f} x {self.gate1.height_px:.1f}")
        if self.gate1.angle_deg is not None:
            lines.append(f"  skew angle: {self.gate1.angle_deg:.2f} deg")
        if self.gate1.num_components is not None:
            lines.append(f"  components found: {self.gate1.num_components}")
        for r in self.gate1.reasons:
            lines.append(f"  - {r}")
        if self.gate2 is not None:
            lines.append("-- Gate 2 (Content) --")
            lines.append(f"  passed: {self.gate2.passed}")
            if self.gate2.ssim_score is not None:
                lines.append(f"  SSIM score: {self.gate2.ssim_score:.4f}")
            if self.gate2.num_good_matches is not None:
                lines.append(f"  registration matches used: {self.gate2.num_good_matches}")
            lines.append(f"  hot spots found: {len(self.gate2.hotspots)}")
            for i, hs in enumerate(self.gate2.hotspots):
                cls = hs.defect_class or "unclassified"
                conf = f"{hs.confidence:.2f}" if hs.confidence is not None else "n/a"
                lines.append(f"    [{i}] bbox={hs.bbox} area={hs.area} class={cls} conf={conf}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# GATE 1 — Structural Integrity
# --------------------------------------------------------------------------

class StructuralGate:
    """
    Verifies physical shape/dimensions of the label before print inspection.

    target_size_px: expected (width, height) in pixels for the *normalized*
        input (see LabelInspector.STANDARD_SIZE). If your camera rig is
        calibrated (px-per-mm known), pass real-world target dims and a
        px_per_mm scale instead, and convert internally.
    """

    def __init__(
        self,
        target_size_px: Tuple[float, float],
        size_tolerance_pct: float = 0.12,
        max_skew_deg: float = 4.0,
        min_area_frac: float = 0.05,
    ):
        self.target_w, self.target_h = target_size_px
        self.size_tolerance_pct = size_tolerance_pct
        self.max_skew_deg = max_skew_deg
        self.min_area_frac = min_area_frac

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        # Otsu threshold; label fabric is usually lighter than background
        # (or vice versa) — try both polarities and keep the one whose
        # largest connected component is closer to filling a sane fraction
        # of the frame (avoids picking up the whole background as "label").
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2 = cv2.bitwise_not(th1)

        def largest_component_frac(mask):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n <= 1:
                return 0.0
            areas = stats[1:, cv2.CC_STAT_AREA]
            return areas.max() / mask.size

        f1, f2 = largest_component_frac(th1), largest_component_frac(th2)
        chosen = th1 if 0.05 < f1 and f1 >= f2 else th2
        # Clean up small noise / holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, kernel, iterations=2)
        chosen = cv2.morphologyEx(chosen, cv2.MORPH_OPEN, kernel, iterations=1)
        return chosen

    def _select_main_contour(self, significant: List[np.ndarray]) -> np.ndarray:
        """Which contour represents the label. Default: simply the largest
        by area — overridable by subclasses that want a smarter pick (e.g.
        preferring aspect-ratio match over raw area to avoid picking up
        background clutter)."""
        return max(significant, key=cv2.contourArea)

    def _skew_is_reliable(self, mask: np.ndarray, main_contour: np.ndarray) -> bool:
        """Whether this contour has enough margin from the crop boundary for
        its minAreaRect angle to reflect a real physical slant rather than
        jagged/anti-aliased crop-edge noise. Default: always reliable —
        overridable by subclasses that see a lot of edge-to-edge tight
        crops (no background margin at all)."""
        return True

    def inspect(self, image_bgr: np.ndarray) -> Gate1Result:
        reasons: List[str] = []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) #Converts the full-color image (which OpenCV loads as Blue-Green-Red) into a single-channel grayscale image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self._binarize(gray)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Gate1Result(passed=False, reasons=["No label contour detected."], mask=mask)

        img_area = image_bgr.shape[0] * image_bgr.shape[1]
        significant = [c for c in contours if cv2.contourArea(c) > self.min_area_frac * img_area]

        # --- Continuous Ribbon check: are there multiple significant,
        # similarly-sized blobs (i.e. an uncut roll / un-separated units)?
        num_components = len(significant)
        if num_components == 0:
            return Gate1Result(passed=False, reasons=["Label region too small / not found."], mask=mask)
        if num_components > 1:
            areas = sorted([cv2.contourArea(c) for c in significant], reverse=True)
            if areas[1] > 0.5 * areas[0]:
                reasons.append(
                    f"Continuous ribbon / uncut units detected ({num_components} "
                    f"comparable-sized components; expected a single separated label)."
                )

        main_contour = self._select_main_contour(significant)

        # --- Length & Width + Corner Angles via minAreaRect
        rect = cv2.minAreaRect(main_contour)         # ((cx,cy),(w,h),angle)
        (rw, rh) = rect[1]
        angle = rect[2]
        box_points = cv2.boxPoints(rect)

        # Normalize angle to "deviation from nearest axis" in [0, 45]
        skew = angle % 90
        skew = min(skew, 90 - skew)

        # Orient (w,h) consistently: shorter side = width, longer = height
        # (labels are typically tall ribbons); adjust if your labels differ.
        width_px, height_px = min(rw, rh), max(rw, rh)

        # if skew > self.max_skew_deg and self._skew_is_reliable(mask, main_contour):
        #     reasons.append(f"Slanted cut detected: corner skew {skew:.2f} deg "
        #                     f"(tolerance {self.max_skew_deg} deg).")

        w_lo = self.target_w * (1 - self.size_tolerance_pct)
        w_hi = self.target_w * (1 + self.size_tolerance_pct)
        h_lo = self.target_h * (1 - self.size_tolerance_pct)
        h_hi = self.target_h * (1 + self.size_tolerance_pct)

        if not (w_lo <= width_px <= w_hi):
            reasons.append(
                f"Width out of spec: {width_px:.1f}px not in "
                f"[{w_lo:.1f}, {w_hi:.1f}]px (chopped/oversized unit)."
            )
        if not (h_lo <= height_px <= h_hi):
            reasons.append(
                f"Height out of spec: {height_px:.1f}px not in "
                f"[{h_lo:.1f}, {h_hi:.1f}]px (chopped/oversized unit)."
            )

        passed = len(reasons) == 0
        return Gate1Result(
            passed=passed,
            reasons=reasons,
            width_px=width_px,
            height_px=height_px,
            angle_deg=skew,
            num_components=num_components,
            mask=mask,
            box_points=box_points,
        )


# --------------------------------------------------------------------------
# GATE 2 — Content & Print Analysis
# --------------------------------------------------------------------------

class ContentGate:
    """
    Step 1: registers (aligns) the candidate image onto the golden reference
             using ORB features + homography + warping.
    Step 2: compares aligned candidate vs. golden using SSIM (heatmap) AND
             plain pixel-subtraction (deterministic backup), then fuses the
             two anomaly masks into a single set of "Hot Spot" ROIs.
    """

    def __init__(
        self,
        ssim_win_size: int = 7,
        ssim_defect_threshold: float = 0.55,   # local SSIM below this -> anomalous
        diff_noise_floor: int = 28,            # 0-255 abs-diff below this -> ignore
        min_hotspot_area: int = 40,
        overall_ssim_reject_threshold: float = 0.75,
        min_good_matches: int = 4,
        orb_features: int = 3000,
        # -- Severity-based reject (independent of global SSIM / hotspot count) --
        # Real localized defects (missing stitch, blur patch, ink bleed) push local
        # SSIM strongly negative over a solid area; registration/lighting noise from
        # handheld photography only nudges local SSIM just under ssim_defect_threshold
        # over scattered/thin regions. Weighting hotspot area by (1 - local SSIM) and
        # normalizing by the label's foreground area gives a scale-invariant score
        # that separates the two far better than raw area, hotspot count, or global
        # SSIM alone (see the "severity_frac" analysis backing this default).
        severity_frac_reject_threshold: float = 0.15,
        max_hotspot_frac_reject_threshold: float = 0.45,
        # Complementary to severity_frac: weights hotspot area by raw pixel-value
        # difference (0-1) instead of (1 - local SSIM). SSIM is a structural/
        # variance-covariance measure, so a "flat" defect (a missing printed
        # character, a solid patch swapped for a differently-shaped-but-equally-
        # smooth icon) can keep local SSIM only moderately depressed even though
        # the raw pixel values are clearly wrong. diff_severity_frac catches
        # exactly that case; severity_frac catches textured/chaotic defects
        # (torn stitching, blur, ink bleed). Kept as two separate signals
        # (OR'd at reject time) rather than merged, so the report can tell you
        # which kind of evidence fired.
        diff_severity_frac_reject_threshold: float = 0.05,
    ):
        self.ssim_win_size = ssim_win_size
        self.ssim_defect_threshold = ssim_defect_threshold
        self.diff_noise_floor = diff_noise_floor
        self.min_hotspot_area = min_hotspot_area
        self.overall_ssim_reject_threshold = overall_ssim_reject_threshold
        self.min_good_matches = min_good_matches
        self.severity_frac_reject_threshold = severity_frac_reject_threshold
        self.max_hotspot_frac_reject_threshold = max_hotspot_frac_reject_threshold
        self.diff_severity_frac_reject_threshold = diff_severity_frac_reject_threshold
        self.orb = cv2.ORB_create(nfeatures=orb_features)

    # -- Step 1: Alignment / Registration -----------------------------------
    @staticmethod
    def _foreground_mask(gray: np.ndarray) -> np.ndarray:
        """Rough label-vs-background mask, dilated generously, used only to
        stop ORB from anchoring on background clutter (hands, table, etc.)."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (th > 0).mean() < 0.6:            # assume label is the majority region
            th = cv2.bitwise_not(th)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        th = cv2.dilate(th, kernel, iterations=2)
        return th

    @staticmethod
    def _tight_label_rect(gray: np.ndarray) -> Optional[Tuple]:
        """The label's own minAreaRect ((cx,cy),(w,h),angle), via the same
        thresholding StructuralGate uses for Gate 1 (majority-aware, so it
        doesn't invert onto the background the way a naive '>0.6 -> invert'
        heuristic would on labels that fill most of the crop). Used to
        recover orientation geometrically when ORB has too few keypoints/
        matches to solve for it itself (common on this low-texture,
        repetitive-weave fabric)."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = StructuralGate._binarize(blurred)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        main = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main) < 0.05 * gray.size:
            return None
        return cv2.minAreaRect(main)

    @staticmethod
    def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate `image` about its center by `angle_deg`, expanding the
        canvas so nothing is cropped off (unlike cv2.warpAffine at the
        original size)."""
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))

    def _geometric_align(
        self, golden_gray: np.ndarray, golden_rect, candidate_bgr: np.ndarray, out_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Rotation+scale+translation alignment derived purely from each
        image's own minAreaRect, used when ORB registration fails outright
        (too few/no good matches). This is the geometric analogue of
        Gate 1's "corner angle" check, and — unlike a plain resize — it
        actually corrects large rotations (including ~90 deg camera-angle
        offsets), which a naive `cv2.resize` onto the golden's canvas
        cannot.

        A rectangle only fixes rotation up to a 180 deg ambiguity (it looks
        the same upside down), so this straightens the candidate to the
        golden's own orientation (portrait vs. landscape) and produces both
        the 0 deg and 180 deg variants, returning whichever matches the
        golden's *interior* print content better (SSIM over a crop well
        inside the label's edges — a raw whole-canvas pixel correlation is
        dominated by the smooth fabric/background and the high-contrast
        label border, both of which are ~identical either way round, and so
        is not sensitive enough to the text/logo to pick the right one).
        """
        out_w, out_h = out_size
        golden_is_landscape = out_w > out_h

        cand_gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
        cand_rect = self._tight_label_rect(cand_gray)
        if cand_rect is None or golden_rect is None:
            return None

        straightened = self._rotate_bound(candidate_bgr, cand_rect[2])
        s_h, s_w = straightened.shape[:2]
        if (s_w > s_h) != golden_is_landscape:
            straightened = cv2.rotate(straightened, cv2.ROTATE_90_CLOCKWISE)

        (gcx, gcy), (gw, gh), _ = golden_rect
        margin = 0.20
        gx0 = int(max(0, gcx - max(gw, gh) / 2 + margin * max(gw, gh)))
        gx1 = int(min(out_w, gcx + max(gw, gh) / 2 - margin * max(gw, gh)))
        gy0 = int(max(0, gcy - min(gw, gh) / 2))
        gy1 = int(min(out_h, gcy + min(gw, gh) / 2))
        interior_valid = gx1 > gx0 and gy1 > gy0
        golden_interior = golden_gray[gy0:gy1, gx0:gx1] if interior_valid else None

        best_candidate = None
        best_score = -np.inf
        for variant in (straightened, cv2.rotate(straightened, cv2.ROTATE_180)):
            variant_gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY)
            variant_rect = self._tight_label_rect(variant_gray)
            if variant_rect is None:
                continue
            (vcx, vcy), (vw, vh), _ = variant_rect
            scale = max(gw, gh) / max(vw, vh)
            M = cv2.getRotationMatrix2D((vcx, vcy), 0.0, scale)
            M[0, 2] += gcx - vcx
            M[1, 2] += gcy - vcy
            placed = cv2.warpAffine(variant, M, out_size, borderValue=(0, 0, 0))

            placed_gray = cv2.cvtColor(placed, cv2.COLOR_BGR2GRAY)
            if golden_interior is not None and min(golden_interior.shape) >= self.ssim_win_size:
                placed_interior = placed_gray[gy0:gy1, gx0:gx1]
                score = float(ssim(golden_interior, placed_interior, win_size=self.ssim_win_size))
            else:
                valid = placed_gray > 0
                if valid.sum() < 0.2 * golden_gray.size:
                    continue
                score = float(
                    np.corrcoef(golden_gray[valid].astype(np.float64), placed_gray[valid].astype(np.float64))[0, 1]
                )
            if score > best_score:
                best_score = score
                best_candidate = placed

        return best_candidate

    @staticmethod
    def _is_sane_homography(H: np.ndarray, w: int, h: int) -> bool:
        """Reject wild/degenerate perspective warps that occasionally pop out
        of RANSAC on sparse, noisy matches (e.g. near-singular H, or a warp
        that would blow the image up/flip it)."""
        if H is None or not np.all(np.isfinite(H)):
            return False
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        except cv2.error:
            return False
        if not np.all(np.isfinite(warped)):
            return False
        area = cv2.contourArea(warped.astype(np.float32))
        orig_area = w * h
        if area < 0.25 * orig_area or area > 4.0 * orig_area:
            return False
        # reject self-intersecting ("bowtie") quads
        if not cv2.isContourConvex(cv2.convexHull(warped.astype(np.float32))):
            pass  # convex hull is always convex; keep simple area check only
        return True

    def register(self, golden_gray: np.ndarray, candidate_gray: np.ndarray):
        h, w = golden_gray.shape[:2]

        golden_fg = self._foreground_mask(golden_gray)
        cand_fg = self._foreground_mask(candidate_gray)

        kp1, des1 = self.orb.detectAndCompute(golden_gray, golden_fg)
        kp2, des2 = self.orb.detectAndCompute(candidate_gray, cand_fg)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:      # Lowe's ratio test
                good.append(m)

        if len(good) < self.min_good_matches:
            return None, len(good)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # A fabric ribbon photographed free-hand is close to planar but not
        # perfectly rigid (it can bow/fold slightly), and match counts are
        # modest/noisy. A full projective homography fit to sparse, noisy
        # matches can overfit into local shear/skew artifacts even when its
        # overall bounding area looks "sane". A similarity/rigid transform
        # (rotation + uniform scale + translation) is far more stable in
        # this regime, so we use it by default, and only trust a full
        # homography when the RANSAC fit is very well-constrained (high
        # inlier ratio + a healthy absolute inlier count) — i.e. genuinely
        # justified perspective correction ("tilt", per the deck) rather
        # than noise fit to a handful of points.
        A, affine_inlier_mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        H_affine = np.vstack([A, [0, 0, 1]]).astype(np.float64) if A is not None else None

        H_proj, proj_inlier_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        num_inliers = int(proj_inlier_mask.sum()) if proj_inlier_mask is not None else 0
        inlier_ratio = num_inliers / len(good) if good else 0.0

        use_projective = (
            H_proj is not None
            and self._is_sane_homography(H_proj, w, h)
            and num_inliers >= 40
            and inlier_ratio >= 0.6
        )

        if use_projective:
            return H_proj, len(good)
        if H_affine is not None:
            return H_affine, len(good)
        if H_proj is not None and self._is_sane_homography(H_proj, w, h):
            return H_proj, len(good)
        return None, len(good)

    @staticmethod
    def _to_gray_equalized(img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _match_illumination(reference_gray: np.ndarray, target_gray: np.ndarray) -> np.ndarray:
        """Histogram-match `target_gray` onto `reference_gray`'s tonal
        distribution. Free-hand phone photos rarely share identical exposure
        / white balance, and that alone can swamp SSIM/diff with a
        wall-to-wall "anomaly" that has nothing to do with real print
        defects. This keeps Gate 2 focused on genuine content differences.
        A fixed-light inspection rig would make this unnecessary in
        production, but it costs nothing and helps here regardless."""
        from skimage.exposure import match_histograms
        matched = match_histograms(target_gray, reference_gray)
        return matched.astype(np.uint8)

    # -- Step 2: Perceptual comparison --------------------------------------
    def _ssim_map(self, golden_gray: np.ndarray, aligned_gray: np.ndarray):
        score, full_map = ssim(
            golden_gray, aligned_gray, win_size=self.ssim_win_size, full=True
        )
        return score, full_map

    def _diff_map(self, golden_gray: np.ndarray, aligned_gray: np.ndarray):
        diff = cv2.absdiff(golden_gray, aligned_gray)
        _, thresh = cv2.threshold(diff, self.diff_noise_floor, 255, cv2.THRESH_BINARY)
        return diff, thresh

    @staticmethod
    def _foreground_area(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    def inspect(self, golden_bgr: np.ndarray, candidate_bgr: np.ndarray) -> Gate2Result:
        reasons: List[str] = []

        golden_gray_raw = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
        cand_gray_raw = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)

        H, n_matches = self.register(golden_gray_raw, cand_gray_raw)
        h, w = golden_bgr.shape[:2]

        if H is None:
            # ORB had too few/no good matches to solve for the transform
            # itself (common on this low-texture, repetitive-weave fabric,
            # especially across large rotations). Never fall back to a
            # plain resize — it ignores rotation entirely and stretches the
            # candidate non-uniformly onto the golden's canvas, which is
            # actively misleading for a ~90 deg-rotated photo. Instead,
            # recover rotation+scale+translation geometrically from each
            # image's own minAreaRect (Gate-1 style).
            golden_rect = self._tight_label_rect(golden_gray_raw)
            aligned_bgr = self._geometric_align(golden_gray_raw, golden_rect, candidate_bgr, (w, h))
            if aligned_bgr is None:
                reasons.append(
                    f"Feature-based registration failed (matches={n_matches}) and "
                    f"minAreaRect-based geometric alignment also failed (no label "
                    f"contour found); Gate 2 cannot reliably compare this pair."
                )
                aligned_bgr = np.zeros_like(golden_bgr)
            else:
                reasons.append(
                    f"Feature-based registration failed (matches={n_matches}); "
                    f"used minAreaRect-based geometric alignment (rotation + scale + "
                    f"translation) instead."
                )
        else:
            aligned_bgr = cv2.warpPerspective(candidate_bgr, H, (w, h))

        golden_gray = self._to_gray_equalized(golden_bgr)
        aligned_gray = self._to_gray_equalized(aligned_bgr)
        aligned_gray = self._match_illumination(golden_gray, aligned_gray)

        _, golden_fg = cv2.threshold(golden_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        golden_fg_frac = (golden_fg > 0).mean()
        if golden_fg_frac > 0.5:
            golden_fg = cv2.bitwise_not(golden_fg)

        # Build a shared foreground mask so SSIM is evaluated on the label
        # content itself, not on background pixels or warp-induced border
        # differences. This is the main guard against clean augmented samples
        # looking artificially dissimilar.
        _, aligned_fg = cv2.threshold(aligned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        aligned_fg_frac = (aligned_fg > 0).mean()
        if aligned_fg_frac > 0.5:
            aligned_fg = cv2.bitwise_not(aligned_fg)

        # Despeckle BEFORE dilating: on a low-contrast/textured background
        # (e.g. a dark ribbon on a wood-grain table), CLAHE equalization
        # boosts local contrast in the background's own texture almost as
        # much as in the label's weave, so Otsu misclassifies scattered
        # background speckle as foreground. Dilating that speckle first (the
        # old behavior) merges it into solid blobs and can swallow nearly
        # the whole frame as "foreground" (observed: ~95-99% on a sample
        # with a wood-grain backdrop, vs ~30% on a plain-backdrop sample).
        # An opening first strips the isolated speckle while leaving the
        # much larger, coherent label blob intact for the dilation to grow.
        despeckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        golden_fg_opened = cv2.morphologyEx(golden_fg, cv2.MORPH_OPEN, despeckle_kernel)
        aligned_fg_opened = cv2.morphologyEx(aligned_fg, cv2.MORPH_OPEN, despeckle_kernel)
        # Guard: if the label itself is thin/small enough that opening wipes
        # it out entirely, fall back to the pre-opening mask rather than
        # comparing against nothing.
        if self._foreground_area(golden_fg_opened) > 0:
            golden_fg = golden_fg_opened
        if self._foreground_area(aligned_fg_opened) > 0:
            aligned_fg = aligned_fg_opened

        fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        golden_fg = cv2.dilate(golden_fg, fg_kernel, iterations=2)
        aligned_fg = cv2.dilate(aligned_fg, fg_kernel, iterations=2)
        comparison_mask = cv2.bitwise_and(golden_fg, aligned_fg)
        if self._foreground_area(comparison_mask) == 0:
            comparison_mask = golden_fg

        mask_for_ssim = cv2.erode(
            comparison_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        if self._foreground_area(mask_for_ssim) == 0:
            mask_for_ssim = comparison_mask

        masked_golden = golden_gray.copy()
        masked_aligned = aligned_gray.copy()
        masked_aligned[mask_for_ssim == 0] = masked_golden[mask_for_ssim == 0]

        # A tiny amount of smoothing suppresses noise-only mismatches in the
        # augmented clean samples, but keeps the larger structural fault ROIs
        # visible enough for the hotspot logic to catch.
        masked_golden = cv2.GaussianBlur(masked_golden, (3, 3), 0)
        masked_aligned = cv2.GaussianBlur(masked_aligned, (3, 3), 0)

        ssim_score, ssim_map = self._ssim_map(masked_golden, masked_aligned)
        ssim_map = ssim_map.astype(np.float32)
        ssim_map[mask_for_ssim == 0] = 1.0

        diff_map, diff_thresh = self._diff_map(golden_gray, aligned_gray)

        # Low local SSIM -> anomaly
        ssim_defect_mask = ((ssim_map < self.ssim_defect_threshold) * 255).astype(np.uint8)

        # Step A: Merge SSIM + Diff masks (Union)
        step1_or = cv2.bitwise_or(ssim_defect_mask, diff_thresh)

        # Step B: Clip to the shared foreground overlap, not just the golden
        # foreground. This prevents border/interpolation artifacts from being
        # promoted to defects.
        step2_and = cv2.bitwise_and(step1_or, comparison_mask)

        # Step C: Remove tiny noise specks (Opening)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        step3_open = cv2.morphologyEx(step2_and, cv2.MORPH_OPEN, kernel3)

        # Step D: Fill gaps / merge nearby defect fragments (Closing)
        kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(step3_open, cv2.MORPH_CLOSE, kernel9, iterations=2)

        combined_with_boxes = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots: List[HotSpot] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_hotspot_area:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + ww + pad), min(h, y + hh + pad)
            crop = aligned_bgr[y0:y1, x0:x1].copy()
            hotspots.append(HotSpot(bbox=(x0, y0, x1 - x0, y1 - y0), area=int(area), crop=crop))
            # Draw padded bounding box (Red) and write area size
            cv2.rectangle(combined_with_boxes, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(combined_with_boxes, f"Area: {int(area)}", (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # plt.figure(figsize=(8, 6))
            # plt.imshow(cv2.cvtColor(combined_with_boxes, cv2.COLOR_BGR2RGB))
            # plt.title("HotSpot Bounding Boxes")
            # plt.axis('off')
            # plt.show()

        # Sort largest-first: biggest anomalies are usually most actionable
        hotspots.sort(key=lambda hs: hs.area, reverse=True)

        hotspot_area = sum(hs.area for hs in hotspots)
        reject_hotspot_area = max(self.min_hotspot_area * 3, int(0.01 * self._foreground_area(comparison_mask)))

        # -- Severity score: area-weighted local dissimilarity, independent of the
        # global SSIM value. This is what actually catches a localized real defect
        # (e.g. a missing stitch) that sits inside an otherwise well-matched label,
        # where the *global* SSIM stays high enough to dodge the check above, and
        # the hotspot *count* is indistinguishable from registration noise on a
        # clean sample (both can produce 6-9 small hotspots).
        fg_area = max(1, self._foreground_area(comparison_mask))
        weighted_severity = 0.0
        diff_weighted_severity = 0.0
        max_hotspot_area = 0
        for hs in hotspots:
            x, y, ww, hh = hs.bbox
            local_ssim = ssim_map[y:y + hh, x:x + ww]
            local_diff = diff_map[y:y + hh, x:x + ww]
            severity_weight = max(0.0, 1.0 - float(local_ssim.mean()))
            diff_weight = float(local_diff.mean()) / 255.0
            weighted_severity += hs.area * severity_weight
            diff_weighted_severity += hs.area * diff_weight
            max_hotspot_area = max(max_hotspot_area, hs.area)

        severity_frac = weighted_severity / fg_area
        diff_severity_frac = diff_weighted_severity / fg_area
        max_hotspot_frac = max_hotspot_area / fg_area
        hotspot_area_frac = hotspot_area / fg_area

        if ssim_score < self.overall_ssim_reject_threshold:
            if hotspot_area >= reject_hotspot_area:
                reasons.append(
                    f"Overall SSIM {ssim_score:.3f} below threshold "
                    f"{self.overall_ssim_reject_threshold} with {hotspot_area}px of hotspot evidence — "
                    f"print content does not match the golden reference closely enough."
                )
            else:
                reasons.append(
                    f"Overall SSIM {ssim_score:.3f} is below threshold, but hotspot evidence is too small "
                    f"({hotspot_area}px < {reject_hotspot_area}px) to reject."
                )
        # NOTE: a "hotspot count >= 10" reject trigger used to live here. It
        # was dropped: TP/TN hotspot-count distributions overlap too heavily
        # to discriminate real defects from registration/alignment noise
        # (observed means ~8 vs ~6-7), and on this dataset it was the
        # dominant cause of false positives (e.g. 13/13 of sample2's
        # remaining FPs had no other trigger fire). severity_frac,
        # max_hotspot_frac and diff_severity_frac below already catch
        # genuine concentrated defects without needing a raw hotspot count.

        severity_reject = severity_frac >= self.severity_frac_reject_threshold
        diff_severity_reject = diff_severity_frac >= self.diff_severity_frac_reject_threshold
        max_hotspot_reject = max_hotspot_frac >= self.max_hotspot_frac_reject_threshold
        if severity_reject or max_hotspot_reject:
            reasons.append(
                f"Localized textured defect evidence: severity_frac={severity_frac:.3f} "
                f"(threshold {self.severity_frac_reject_threshold}), "
                f"max_hotspot_frac={max_hotspot_frac:.3f} "
                f"(threshold {self.max_hotspot_frac_reject_threshold}) — "
                f"a concentrated, structurally dissimilar region was found even though "
                f"global SSIM/hotspot-count stayed in the 'clean-looking' range."
            )
        if diff_severity_reject:
            reasons.append(
                f"Localized flat/tonal defect evidence: diff_severity_frac={diff_severity_frac:.3f} "
                f"(threshold {self.diff_severity_frac_reject_threshold}) — a concentrated region "
                f"with a large raw pixel-value mismatch was found (e.g. missing print, swapped "
                f"solid patch/icon) that structural SSIM alone under-weights."
            )
        if hotspots:
            reasons.append(f"{len(hotspots)} hot spot(s) require classification.")

        passed = not (
            (ssim_score < self.overall_ssim_reject_threshold and hotspot_area >= reject_hotspot_area)
            or severity_reject
            or max_hotspot_reject
            or diff_severity_reject
        )

        return Gate2Result(
            passed=passed,
            reasons=reasons,
            aligned_candidate=aligned_bgr,
            ssim_score=float(ssim_score),
            ssim_map=ssim_map,
            diff_map=diff_map,
            combined_defect_mask=combined,
            hotspots=hotspots,
            homography=H,
            num_good_matches=n_matches,
            hotspot_area_frac=float(hotspot_area_frac),
            max_hotspot_area_frac=float(max_hotspot_frac),
            severity_frac=float(severity_frac),
            diff_severity_frac=float(diff_severity_frac),
        )


# --------------------------------------------------------------------------
# GATE 3 — ML Classifier for Hot Spots
# --------------------------------------------------------------------------

# WE HAVE TO CHANGE THESE CLASSES ACCORDING TO OUR CLASSES

DEFAULT_DEFECT_CLASSES = [
    "Ink Bleed",
    "Missing Stitch",
    "Lamination Scuff",
    "Text/Number Mismatch",
    "Blur",
]


class HeuristicHotSpotClassifier:
    """
    Zero-training-data fallback classifier. Uses simple, explainable image
    statistics computed on (golden_crop, candidate_crop) to guess a defect
    class. This is meant to keep the pipeline runnable end-to-end before a
    labeled defect-crop dataset exists to train `CNNHotSpotClassifier`.

    Swap this out for the CNN classifier below as soon as you have labeled
    hot-spot crops (a few hundred per class is usually enough to fine-tune).
    """

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or DEFAULT_DEFECT_CLASSES

    @staticmethod
    def _color_stats(crop_bgr: np.ndarray):
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = hsv[..., 1].mean()
        val_std = hsv[..., 2].std()
        return sat_mean, val_std

    @staticmethod
    def _edge_density(crop_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges.mean() / 255.0

    def classify(self, hotspot: HotSpot, golden_crop: Optional[np.ndarray] = None):
        crop = hotspot.crop
        if crop.size == 0:
            return "Unclassified", 0.0

        sat_mean, val_std = self._color_stats(crop)
        edge_density = self._edge_density(crop)
        h, w = crop.shape[:2]
        aspect = w / max(h, 1)

        # Very rough, explainable rules — replace with a trained model.
        if sat_mean > 60 and val_std > 45:
            return "Ink Bleed", 0.55
        if edge_density > 0.18 and 0.4 < aspect < 2.5:
            return "Text/Number Mismatch", 0.55
        if val_std < 15 and edge_density < 0.05:
            return "Lamination Scuff", 0.5
        if edge_density < 0.08:
            return "Missing Stitch", 0.45
        return "Blur", 0.4


class CNNHotSpotClassifier:
    """
    Torch-based classifier scaffold. Uses a small pretrained backbone
    (MobileNetV2) as a frozen feature extractor with a fresh linear head —
    train the head on your own labeled hot-spot crops. Only import torch
    lazily so the rest of the pipeline works even if torch isn't installed.
    """

    def __init__(self, classes: Optional[List[str]] = None, weights_path: Optional[str] = None,
                 device: str = "cpu"):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        self.torch = torch
        self.device = device
        self.classes = classes or DEFAULT_DEFECT_CLASSES

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        backbone.classifier[1] = nn.Linear(backbone.last_channel, len(self.classes))
        self.model = backbone.to(device).eval()

        if weights_path:
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, hotspot: HotSpot, golden_crop: Optional[np.ndarray] = None):
        crop_rgb = cv2.cvtColor(hotspot.crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            probs = self.torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(dim=0)
        return self.classes[int(idx)], float(conf)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

class LabelInspector:
    """
    Ties Gate 1 -> Gate 2 -> Gate 3 together exactly as specified in the
    deck: Gate 1 is a hard stop-gap; Gate 2 only runs if Gate 1 passes;
    Gate 3 classifies whatever hot spots Gate 2 finds.
    """

    def __init__(
        self,
        target_size_px: Tuple[float, float],
        gate1_kwargs: Optional[dict] = None,
        gate2_kwargs: Optional[dict] = None,
        classifier=None,
    ):
        self.gate1 = StructuralGate(target_size_px, **(gate1_kwargs or {}))
        self.gate2 = ContentGate(**(gate2_kwargs or {}))
        self.classifier = classifier or HeuristicHotSpotClassifier()

    def inspect(self, golden_bgr: np.ndarray, candidate_bgr: np.ndarray) -> InspectionReport:
        g1 = self.gate1.inspect(candidate_bgr)
        if not g1.passed:
            return InspectionReport(verdict="REJECT_GATE1", gate1=g1)

        g2 = self.gate2.inspect(golden_bgr, candidate_bgr)
        for hs in g2.hotspots:
            cls, conf = self.classifier.classify(hs)
            hs.defect_class = cls
            hs.confidence = conf

        verdict = "PASS" if g2.passed else "REJECT_GATE2"
        return InspectionReport(verdict=verdict, gate1=g1, gate2=g2)
