"""Recognize the visible score and replay control in Jump Jump game screenshots."""
import cv2
import numpy as np

BASE_W = 543
BASE_H = 1029

# Each cell is one square in the game's bitmap typeface. 1 is half-width.
GLYPHS = {
    0: ("1111", "1001", "1001", "1001", "1111"),
    1: ("11", "01", "01", "01", "01"),
    2: ("1111", "0001", "1111", "1000", "1111"),
    3: ("1111", "0001", "1111", "0001", "1111"),
    4: ("1001", "1001", "1001", "1111", "0001"),
    5: ("1111", "1000", "1111", "0001", "1111"),
    6: ("1110", "1000", "1111", "1001", "1111"),
    7: ("1111", "0001", "0010", "0100", "0100"),
    8: ("1111", "1001", "1111", "1001", "1111"),
    9: ("1111", "1001", "1111", "0001", "0111"),
}
BITS = {tuple(map(int, "".join(rows))): digit for digit, rows in GLYPHS.items()}


def _rgb(frame):
    if frame is None or not hasattr(frame, "convert"):
        return None
    if frame.width < BASE_W // 2 or frame.height < BASE_H // 2:
        return None
    if abs((frame.width / frame.height) / (BASE_W / BASE_H) - 1) > 0.04:
        return None
    data = np.asarray(frame.convert("RGB"))
    if data.shape[:2] != (BASE_H, BASE_W):
        data = cv2.resize(data, (BASE_W, BASE_H), interpolation=cv2.INTER_AREA)
    return data


def read_score(frame):
    """Return a verified numeric HUD score or None if the HUD is absent/unclear."""
    data = _rgb(frame)
    if data is None:
        return None
    # All ten digits contain the upper-left cell. Sample its interior and
    # the adjacent background so translucent HUD text works against each theme.
    crop = data[160:222, 55:400].astype(np.float32)
    ink = np.median(crop[12:16, 11:15], axis=(0, 1))
    background = np.median(crop[:, :6], axis=1, keepdims=True)
    direction = ink - background
    contrast = np.sum(direction ** 2, axis=2)
    if np.median(contrast) < 45 ** 2:
        return None
    coverage = np.sum((crop - background) * direction, axis=2) / np.maximum(contrast, 1)
    mask = np.uint8(coverage > 0.52)
    # Diagonals can meet only at their corners (notably the 7). Split digits
    # by their column projections so antialiasing does not fragment them.
    occupied = np.any(mask, axis=0)
    edges = np.diff(np.r_[0, occupied, 0].astype(np.int8))
    components = []
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        glyph = mask[:, left:right]
        rows = np.flatnonzero(glyph.any(axis=1))
        if glyph.sum() >= 25:
            components.append((left, rows[0], right - left, rows[-1] - rows[0] + 1))
    if not components or len(components) > 7:
        return None
    if abs(components[0][0] + 55 - 64) > 3:
        return None
    score = 0
    end = None
    for x, y, w, h in components:
        if abs(y + 160 - 169) > 3 or abs(h - 42) > 3:
            return None
        if end is not None and not (2 <= x - end <= 27):
            return None
        end = x + w
        columns = round(w / (h / 5))
        if columns not in (2, 4) or abs(w - columns * h / 5) > 2.5:
            return None
        glyph = mask[y:y+h, x:x+w].astype(np.float32)
        cells = cv2.resize(glyph, (columns, 5), interpolation=cv2.INTER_AREA)
        if np.any((cells > 0.22) & (cells < 0.78)):
            return None
        digit = BITS.get(tuple((cells > 0.5).astype(int).ravel()))
        if digit is None:
            return None
        score = score * 10 + digit
    return score


def restart_button(frame):
    """Return the native-pixel center of the confirmed replay button, else None."""
    data = _rgb(frame)
    if data is None:
        return None
    # End screens show a dark background, a bright score and a white title above it.
    title = data[162:187, 204:332, :].min(axis=2) > 180
    score = data[216:294, 70:473, :].min(axis=2) > 218
    if not (70 < int(title.sum()) < 1600 and 520 < int(score.sum()) < 15000):
        return None
    if data[302:320, 25:80].mean() >= 174:
        return None
    # Locate the circular green arrow and identify its surrounding white button
    # and four dark text characters to the right. Ignore the advertisement below.
    hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv[535:888, 85:445], (36, 122, 122), (99, 255, 255))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y = x + 85, y + 535
        if not (19 <= w <= 36 and 20 <= h <= 36 and 0.65 < w/h < 1.6):
            continue
        cy = y + h // 2
        right = data[cy - 19:cy + 20, x + w + 4:x + w + 131]
        left = data[cy - 11:cy + 12, x - 23:x - 4]
        if left.size == 0 or right.shape[1] < 115:
            continue
        if (left.min(axis=2) > 234).mean() < 0.91:
            continue
        if (right.min(axis=2) > 232).mean() < 0.52:
            continue
        ink = (right.max(axis=2) < 97)
        if not (420 <= ink.sum() <= 1850):
            continue
        # Detect multiple characters, including final character well past the icon.
        if any(ink[:, a:b].sum() < 35 for a, b in ((3,26),(28,50),(52,76),(80,110))):
            continue
        return (round((x + w + 58) * frame.width / BASE_W),
                round(cy * frame.height / BASE_H))
    return None
