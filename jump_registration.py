"""Read-only camera translation from stationary game scenery in two 543px frames."""
import cv2
import numpy as np


def _static_corners(gray, foot, target):
    """Locate separated corners on platforms, avoiding the piece and its effects."""
    foot_x, foot_y = foot
    target_x, target_y = target
    mask = np.zeros_like(gray)
    mask[400:875, 17:526] = 255
    mask[max(0, foot_y - 133):foot_y + 24, max(0, foot_x - 80):foot_x + 54] = 0
    mask[target_y - 20:target_y + 24, max(0, target_x - 22):target_x + 23] = 0
    return cv2.goodFeaturesToTrack(gray, 36, .008, 30, mask=mask, blockSize=5)


def _match_patch(first, second, origin, guess, landed_foot):
    x, y = origin
    half = 12
    search_radius = 50
    predicted_x, predicted_y = origin + guess
    left = max(2, predicted_x - search_radius - half)
    right = min(541, predicted_x + search_radius + half + 1)
    top = max(320, predicted_y - search_radius - half)
    bottom = min(964, predicted_y + search_radius + half + 1)
    if right - left < 2 * half + 1 or bottom - top < 2 * half + 1:
        return None
    patch = first[y - half:y + half + 1, x - half:x + half + 1]
    scores = cv2.matchTemplate(second[top:bottom, left:right], patch, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (match_x, match_y) = cv2.minMaxLoc(scores)
    if peak < .953:
        return None
    dest_x, dest_y = left + match_x + half, top + match_y + half
    foot_x, foot_y = landed_foot
    if foot_x - 75 <= dest_x <= foot_x + 51 and foot_y - 128 <= dest_y <= foot_y + 22:
        return None
    # Exclude the winning extremum and reject a nearly interchangeable second match.
    scores[max(0, match_y - 5):match_y + 6, max(0, match_x - 5):match_x + 6] = -1
    if peak - cv2.minMaxLoc(scores)[1] < .028:
        return None
    return (dest_x - x, dest_y - y), peak


def fallback_shift(first, second, before_foot, target, after_foot):
    """Return (numpy shift, evidence) or None. All input points use the 543px view.

    Intended for settled endpoints of one confirmed jump, following an absent
    or geometrically invalid primary registration. The expected landing point
    only bounds the image search. The returned shift is estimated from scenery.
    The caller still checks score gain, the target and the lateral residual.
    """
    before = np.asarray(first.convert("L"))
    after = np.asarray(second.convert("L"))
    if before.shape != (1029, 543) or after.shape != before.shape:
        return None
    guess = np.rint(np.asarray(after_foot) - target).astype(int)
    corners = _static_corners(before, before_foot, target)
    if corners is None:
        return None
    origins, moves, scores = [], [], []
    for corner in corners[:, 0, :]:
        origin = np.rint(corner).astype(int)
        match = _match_patch(before, after, origin, guess, after_foot)
        if match is not None:
            move, score = match
            origins.append(origin)
            moves.append(move)
            scores.append(score)
    if len(moves) < 7:
        return None
    moves = np.asarray(moves)
    disagreements = np.max(np.abs(moves[:, None, :] - moves[None, :, :]), axis=2)
    groups = disagreements <= 2
    selected = groups[np.argmax(groups.sum(axis=1))]
    if selected.sum() < max(7, .55 * len(moves)):
        return None
    positions = np.asarray(origins)[selected]
    extent = np.ptp(positions, axis=0)
    span = np.linalg.norm(extent)
    hull_area = cv2.contourArea(cv2.convexHull(positions.astype(np.float32)))
    if span < 98 or extent.min() < 35 or hull_area < 2000:
        return None
    # At least two distant details each need another corroborating corner.
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    supported = ((distances > 0) & (distances <= 80)).any(axis=1)
    if not np.any((distances > 105) & supported[:, None] & supported[None, :]):
        return None
    shift = np.median(moves[selected], axis=0)
    if np.linalg.norm(shift) < 8:
        return None
    residuals = np.linalg.norm(moves[selected] - shift, axis=1)
    quality = dict(
        method="template_consensus",
        matches=len(moves),
        inliers=int(selected.sum()),
        inlier_ratio=round(float(selected.mean()), 4),
        feature_span_px=round(float(span), 2),
        median_inlier_residual_px=round(float(np.median(residuals)), 3),
        area_px2=round(hull_area, 1),
        min_inlier_ncc=round(float(np.min(np.asarray(scores)[selected])), 4),
    )
    return shift, quality
