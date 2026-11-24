"""Play the foreground WeChat Jump game. Distances use the 543x1029 view."""
import argparse
import ctypes
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from jump_early_landing import ready_pair
from types import ModuleType
from ctypes import wintypes as wt
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageGrab

ROOT = Path(__file__).resolve().parent
SIZE = (543, 1029)
POINTS = ((65, .230), (102, .337), (125, .4189), (170, .5506), (220, .7107), (268, .8429))


def baseline_duration(length):
    for i in range(1, len(POINTS)):
        if length <= POINTS[i][0]:
            break
    if length > POINTS[-1][0]:
        return POINTS[-1][1] + (length - POINTS[-1][0]) / 320
    left, right = POINTS[i - 1], POINTS[i]
    return left[1] + (length - left[0]) * (right[1] - left[1]) / (right[0] - left[0])


BASELINE_VERSION = "curve-" + hashlib.sha256(
    repr((POINTS, baseline_duration.__code__.co_consts)).encode() + baseline_duration.__code__.co_code
).hexdigest()[:12]


def timing_prediction(length, model):
    if model is not None:
        try:
            result = model.predict_hold(length)
            if result is not None and math.isfinite(result) and .12 <= result <= 1.12:
                return float(result), "model"
        except (ValueError, TypeError, ArithmeticError):
            pass
    return baseline_duration(length), "baseline"


def along(dx, dy, direction):
    return (direction * dx - math.sqrt(3) * dy) / 2


class GameWindow:
    def __init__(self, hwnd, background=False):
        self.background = background
        self.user = ctypes.windll.user32
        self.user.SetProcessDPIAware()
        if hwnd is None:
            self.user.FindWindowW.argtypes = (wt.LPCWSTR, wt.LPCWSTR)
            self.user.FindWindowW.restype = wt.HWND
            hwnd = self.user.FindWindowW(None, "跳一跳")
            if not hwnd:
                raise RuntimeError("Open the WeChat 跳一跳 game window first.")
        self.hwnd = hwnd
        self.user.GetForegroundWindow.restype = wt.HWND
        self.user.GetAncestor.argtypes = (wt.HWND, wt.UINT)
        self.user.GetAncestor.restype = wt.HWND
        self.user.WindowFromPoint.argtypes = (wt.POINT,)
        self.user.WindowFromPoint.restype = wt.HWND
        self.user.ScreenToClient.argtypes = (wt.HWND, ctypes.POINTER(wt.POINT))
        self.user.PostMessageW.argtypes = (wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
        self.user.PostMessageW.restype = wt.BOOL
        self.user.GetWindowTextW.argtypes = (wt.HWND, wt.LPWSTR, ctypes.c_int)
        dwm = ctypes.windll.dwmapi
        self.bounds = dwm.DwmGetWindowAttribute
        self.bounds.argtypes = (wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD)
        self.bounds.restype = ctypes.c_long
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
        kernel.OpenProcess.restype = wt.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = (wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD))
        kernel.CloseHandle.argtypes = (wt.HANDLE,)
        self.user.GetWindowThreadProcessId.argtypes = (wt.HWND, ctypes.POINTER(wt.DWORD))
        self.pid = wt.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(self.pid))
        handle = kernel.OpenProcess(0x1000, False, self.pid)
        if not handle:
            raise RuntimeError("Cannot verify game process")
        try:
            name = ctypes.create_unicode_buffer(4096)
            count = wt.DWORD(len(name))
            valid = kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(count))
            if not valid or Path(name.value).name.lower() != "wechatappex.exe":
                raise RuntimeError("Handle belongs to a different process")
        finally:
            kernel.CloseHandle(handle)

    def check(self):
        title = ctypes.create_unicode_buffer(256)
        self.user.GetWindowTextW(self.hwnd, title, len(title))
        if title.value != "跳一跳" or not self.user.IsWindowVisible(self.hwnd) or self.user.IsIconic(self.hwnd):
            raise RuntimeError("Jump window is unavailable or minimized")
        if not self.background and self.user.GetForegroundWindow() != self.hwnd:
            raise RuntimeError("Jump window is no longer in the foreground")
        rect = wt.RECT()
        if self.bounds(self.hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)):
            raise RuntimeError("Cannot read physical window bounds")
        bbox = rect.left, rect.top, rect.right, rect.bottom
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 300 or height < 500 or abs(width / height - SIZE[0] / SIZE[1]) > .025:
            raise RuntimeError(f"Unexpected window geometry {bbox}")
        return bbox

    def point(self, bbox, xy):
        left, top, right, bottom = bbox
        x = left + round(xy[0] * (right - left) / SIZE[0])
        y = top + round(xy[1] * (bottom - top) / SIZE[1])
        owner = self.user.WindowFromPoint(wt.POINT(x, y))
        if self.user.GetAncestor(owner, 2) != self.hwnd:
            raise RuntimeError(f"Window is covered at {xy}")
        return x, y

    def frame(self):
        bbox = self.check()
        for xy in ((80, 190), (270, 425), (110, 580), (415, 580), (270, 765)):
            self.point(bbox, xy)
        from jump_capture import grab_rect
        try:
            frame = grab_rect(bbox)
        except OSError:
            frame = ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")
        if self.check() != bbox:
            raise RuntimeError("Window moved during capture")
        if frame.size != SIZE:
            frame = frame.resize(SIZE, Image.Resampling.BILINEAR)
        return frame, bbox

    def hold(self, bbox, xy, seconds):
        if self.check() != bbox:
            raise RuntimeError("Window moved after observation")
        point = self.point(bbox, xy)
        if self.background:
            screen = wt.POINT(*point)
            owner = self.user.WindowFromPoint(screen)
            if self.user.GetAncestor(owner, 2) != self.hwnd:
                raise RuntimeError("Game input location became covered")
            client = wt.POINT(*point)
            if not self.user.ScreenToClient(owner, ctypes.byref(client)):
                raise RuntimeError("Cannot translate game input coordinates")
            packed = ((client.y & 0xffff) << 16) | (client.x & 0xffff)
            self.last_press = time.monotonic()
            if not self.user.PostMessageW(owner, 0x0201, 1, packed):
                raise RuntimeError("Game did not accept mouse-down message")
            start = time.perf_counter()
            try:
                time.sleep(seconds)
            finally:
                if not self.user.PostMessageW(owner, 0x0202, 0, packed):
                    raise RuntimeError("Game did not accept mouse-up message")
            return time.perf_counter() - start
        if self.user.GetAsyncKeyState(1) & 0x8000:
            raise RuntimeError("Mouse is already held")
        original = wt.POINT()
        if not self.user.GetCursorPos(ctypes.byref(original)):
            raise RuntimeError("Cannot read pointer position")
        down = False
        try:
            if not self.user.SetCursorPos(*point):
                raise RuntimeError("Cannot move pointer")
            self.point(bbox, xy)
            if self.check() != bbox:
                raise RuntimeError("Window moved before input")
            self.last_press = time.monotonic()
            self.user.mouse_event(0x0002, 0, 0, 0, 0)
            down = True
            start = time.perf_counter()
            end = start + seconds
            while (remaining := end - time.perf_counter()) > 0:
                time.sleep(min(.02, remaining))
                if self.user.GetForegroundWindow() != self.hwnd:
                    raise RuntimeError("Foreground changed during input")
        finally:
            if down:
                self.user.mouse_event(0x0004, 0, 0, 0, 0)
            now = wt.POINT()
            self.user.GetCursorPos(ctypes.byref(now))
            if (now.x, now.y) == point:
                self.user.SetCursorPos(original.x, original.y)
        return time.perf_counter() - start


def camera_shift(first, second, with_quality=False):
    # Stable edges on platforms move with the camera. Require agreement between distinct corners.
    orb = cv2.ORB_create(nfeatures=450, fastThreshold=9)
    def features(image):
        band = np.asarray(image.convert("L"))[360:880]
        return orb.detectAndCompute(band, None)
    a, da = features(first)
    b, db = features(second)
    if da is None or db is None or len(db) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .70 * p[1].distance]
    if len(good) < 7:
        return None
    origins = np.array([a[m.queryIdx].pt for m in good])
    moves = np.array([b[m.trainIdx].pt for m in good]) - origins
    groups = np.max(np.abs(moves[:, None] - moves[None, :]), axis=2) < 4
    chosen = groups[np.argmax(groups.sum(axis=1))]
    if sum(chosen) < max(7, .48 * len(good)) or np.linalg.norm(np.ptp(origins[chosen], axis=0)) < 58:
        return None
    shift = np.median(moves[chosen], axis=0)
    if not with_quality:
        return shift
    quality = {"method": "ORB_translation", "matches": len(good), "inliers": int(sum(chosen)),
               "inlier_ratio": round(float(np.mean(chosen)), 4),
               "feature_span_px": round(float(np.linalg.norm(np.ptp(origins[chosen], axis=0))), 2),
               "median_inlier_residual_px": round(float(np.median(np.linalg.norm(moves[chosen]-shift, axis=1))), 3)}
    return shift, quality


def inspect_geometry(scene):
    foot, target = scene.get("foot"), scene.get("target")
    if not foot or not target or scene.get("confidence", 0) < .78:
        return None
    dx, dy = target[0] - foot[0], target[1] - foot[1]
    direction = 1 if dx > 0 else -1
    length = along(dx, dy, direction)
    if not (25 <= length <= 400 and dy < -15 and abs(dx) > 22):
        return None
    if abs(abs(dx) + math.sqrt(3) * dy) / 2 > 58:
        return None
    return direction, length


def registered_landing(previous, frame, scene, score, changed, fallback):
    gain = score - previous["score"] if score is not None and previous["score"] is not None else None
    def evaluate(result):
        if result is None:
            return None
        shift, quality = result
        fx, fy = scene["foot"]
        px, py = previous["foot"]
        s = previous["direction"]
        actual = along(fx-px-shift[0], fy-py-shift[1], s)
        dx, dy = fx-shift[0]-previous["target"][0], fy-shift[1]-previous["target"][1]
        lateral = abs(s*dx + math.sqrt(3)*dy)/2
        rejected = []
        if not 25 <= actual <= 455 or abs(actual-previous["length"]) >= 55:
            rejected.append("axis")
        if lateral > 12:
            rejected.append("lateral")
        if (quality["inliers"] < 7 or quality["inlier_ratio"] < .48 or quality["feature_span_px"] < 58
                or quality["median_inlier_residual_px"] > 3):
            rejected.append("registration")
        quality = {**quality, "lateral_error_px": round(lateral, 3), "rejected": rejected}
        return shift, actual, quality
    primary = evaluate(camera_shift(previous["frame"], frame, with_quality=True))
    if primary is not None and not primary[2]["rejected"]:
        return primary
    if fallback is None or gain is None or gain <= 0 or not changed or previous["confidence"] < .8:
        return primary
    candidate = evaluate(fallback(previous["frame"], frame, previous["foot"], previous["target"], scene["foot"]))
    if candidate is not None and not candidate[2]["rejected"]:
        candidate[2]["primary"] = primary[2] if primary is not None else {"method": "ORB_translation", "rejected": ["no_fit"]}
        return candidate
    return primary


def short_aim(length, direction, counts, enabled=True):
    if not enabled or length >= 105:
        return None, None, 0.
    band = 0 if length < 85 else 1 if length < 95 else 2
    stratum = f"{direction}/{band}"
    index = counts.get(stratum, 0)
    treatment = (index + band + int(direction > 0)) % 2 == 1
    offset = -1.2 * min(1., (105-length)/10) if treatment else 0.
    return stratum, "shorten" if treatment else "control", offset


def prior_aim_counts(field="aim_stratum"):
    counts = {}
    for path in (ROOT / "runs").glob("*/events.jsonl"):
        with path.open(encoding="utf-8") as log:
            for line in log:
                if f'"{field}"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = row.get(field)
                if key and row.get("event") == "jump":
                    counts[key] = counts.get(key, 0) + 1
    return counts


def marker_aim(scene, geometry, kind, wait, counts, enabled):
    raw, point = scene.get("target"), scene.get("marker_target")
    if not enabled or raw is None or point is None or geometry is None or kind is not None:
        return {}
    direction, length = geometry
    if (not 110 <= length <= 280 or scene.get("confidence", 0) < .93
            or scene.get("reason") != "top material outline" or scene.get("marker_updated") is not True):
        return {}
    dx, dy = point[0]-raw[0], point[1]-raw[1]
    if not all(math.isfinite(v) for v in (dx,dy)) or abs(dx) > 6 or abs(dy) > 8:
        return {}
    stratum = f"{direction}/{wait}"
    use_marker = (counts.get(stratum, 0) + int(direction > 0)) % 2 == 1
    proposed = scene.get("marker_aim_target") or point
    dx_used = float(np.clip(proposed[0]-raw[0], -2., 2.)) if use_marker else 0.
    return dict(marker_group="marker_x" if use_marker else "geometry", marker_stratum=stratum,
                marker_raw_target=list(raw), marker_target=list(point), marker_aim_target=[raw[0]+dx_used, raw[1]],
                marker_confidence=scene["marker_confidence"], marker_version=scene.get("marker_version"),
                marker_axis_offset=round(direction*dx_used/2, 4))


def aim_duration_delta(length, offset, model, source, base):
    if not offset:
        return 0.
    adjusted, chosen = timing_prediction(length + offset, model)
    if chosen == source:
        return adjusted - base
    # A treatment straddling the calibrated domain must use the same model as its control.
    opposite, chosen = timing_prediction(length - offset, model)
    return base - opposite if chosen == source else 0.


def refresh_readers(readers, record):
    for name, required in (("jump_vision", ("analyze",)), ("jump_score", ("read_score", "restart_button")),
                           ("jump_bonus", ("classify_target",)), ("jump_timing", ("predict_hold",)),
                           ("jump_registration", ("fallback_shift",))):
        path = ROOT / (name + ".py")
        try:
            stat = path.stat()
            version = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            if name in ("jump_bonus", "jump_timing", "jump_registration") or name in readers:
                continue
            raise
        old = readers.get(name)
        if old and old[0] == version:
            continue
        try:
            module = ModuleType(name)
            module.__file__ = str(path)
            source = path.read_bytes()
            exec(compile(source, str(path), "exec"), module.__dict__)
            module._source_version = "sha256-" + hashlib.sha256(source).hexdigest()[:16]
            if not all(callable(getattr(module, fn, None)) for fn in required):
                raise ValueError("reader interface is incomplete")
        except Exception as exc:
            record("reader_update_deferred", module=name, error=str(exc))
            if not old and name not in ("jump_bonus", "jump_timing", "jump_registration"):
                raise RuntimeError(f"{name} is unavailable") from exc
            continue
        sys.modules[name] = module
        readers[name] = (version, module)
        record("reader_loaded", module=name, modified_ns=version[0],
               version=version_of(module) if name == "jump_timing" else module._source_version)


def version_of(module):
    if module is None:
        return BASELINE_VERSION
    for field in ("VERSION", "MODEL_VERSION", "_source_version"):
        value = getattr(module, field, None)
        if value is not None:
            return str(value)
    raise ValueError("Model has no source fingerprint")


class OnlineTraining:
    """Start one finite training job after each batch of newly measured jumps."""
    def __init__(self, stop_file, record):
        self.stop_file = stop_file
        self.record = record
        self.pending = self.unsubmitted()
        self.proc = None
        self.log = None
        self.closed = False
        self.missing = False

    @staticmethod
    def unsubmitted():
        count = 0
        for path in sorted((ROOT / "runs").glob("*/events.jsonl"), reverse=True):
            try:
                with path.open(encoding="utf-8") as log:
                    lines = log.readlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item.get("event") == "training_started":
                    return count
                if item.get("event") == "calibration" and item.get("eligible") is True:
                    count += 1
                    if count >= 16:
                        return count
        return count

    def finish(self):
        code = self.proc.returncode
        self.log.seek(0, os.SEEK_END)
        size = self.log.tell()
        self.log.seek(max(0, size - 12000))
        lines = self.log.read().decode("utf-8", "replace").splitlines()
        self.log.close()
        self.proc = self.log = None
        outcome = lines[-1] if lines else ""
        try:
            outcome = json.loads(outcome)
        except ValueError:
            pass
        self.record("training_finished", exit_code=code, outcome=outcome)

    def tick(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.finish()
        if self.closed or self.stop_file.exists() or self.proc is not None or self.pending < 16:
            return
        script = ROOT / "jump_online.py"
        if not script.is_file():
            if not self.missing:
                self.record("training_deferred", reason="trainer_not_ready", pending=self.pending)
            self.missing = True
            return
        self.missing = False
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if self.stop_file.exists():
            return
        self.log = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen([sys.executable, str(script), "--once"], cwd=ROOT,
                                         stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                         env=env, creationflags=flags)
        except OSError as exc:
            self.log.close()
            self.log = None
            self.record("training_deferred", reason=str(exc), pending=self.pending)
            self.pending = 0
            return
        self.record("training_started", pid=self.proc.pid, new_samples=self.pending)
        self.pending = 0

    def close(self):
        self.closed = True
        if self.proc is None:
            return
        if self.proc.poll() is None:
            if self.stop_file.exists():
                self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.finish()


def falling_pawn(frame, target):
    # A toppled piece has its purple head beside its body; a standing piece has the head above it.
    rgb = np.asarray(frame.convert("RGB"), dtype=np.int16)
    r, g, b = rgb.transpose(2, 0, 1)
    mask = (b-r > 8) & (b-g > 6) & (r < 115) & (g < 110) & (b < 155)
    mask[:360] = False
    mask[840:] = False
    parts = cv2.connectedComponentsWithStats(mask.astype(np.uint8))[2][1:]
    for hx, hy, hw, hh, ha in parts:
        if not (19 <= hw <= 35 and 19 <= hh <= 35 and 250 <= ha <= 1020):
            continue
        cx, cy = hx + hw/2, hy + hh/2
        if abs(cx-target[0]) > 160 or not -100 < cy-target[1] < 205:
            continue
        for x, y, w, h, a in parts:
            if 550 < a < 3400 and 25 <= w <= 91 and 20 <= h <= 83:
                dx, dy = abs(cx-x-w/2), abs(cy-y-h/2)
                if 17 <= dx <= 70 and math.hypot(dx, dy) <= 75:
                    return True
    return False


def wait_for_landing(game, previous, stop_file, collect, *, delay=1.30):
    release = previous["release"]
    frames = []
    for offset in ((.22, .50, .85, delay) if collect else (delay,)):
        while (remaining := release + offset - time.monotonic()) > 0 and not stop_file.exists():
            time.sleep(min(.05, remaining))
        if stop_file.exists() or offset == delay:
            break
        try:
            shot, _ = game.frame()
        except (RuntimeError, OSError):
            continue
        frames.append((round(time.monotonic() - release, 3), shot))
    return frames


def early_observation(game, previous, readers):
    started = time.monotonic()
    frame, bbox = game.frame()
    when = (started + time.monotonic()) / 2 - previous["release"]
    scores = readers["jump_score"][1]
    score = scores.read_score(frame)
    if score is None or score <= previous["score"] or scores.restart_button(frame) is not None:
        return None
    # The empty band above the board is clear in established game scenes.
    band = np.asarray(frame)[252:327:9, 8:-8:8].astype(np.int16)
    noise = np.abs(band-np.median(band, axis=1, keepdims=True)).max(axis=2)
    if np.mean(noise > 15) > .025:
        return None
    scene = readers["jump_vision"][1].analyze(frame)
    geom = inspect_geometry(scene)
    bonus = readers.get("jump_bonus", (None, None))[1]
    if (not geom or geom[1] < 105 or scene["confidence"] < .93 or bonus is None
            or bonus.classify_target(frame, scene) is not None):
        return None
    return (when, score, scene), (frame, bbox)


def landing_trial(game, previous, stop_file, readers, group):
    """Evaluate identical observations in both arms of the alternating trial."""
    early = None
    observations = []
    for at in (1., 1.12):
        wait_for_landing(game, previous, stop_file, False, delay=at)
        if stop_file.exists():
            break
        try:
            observation = early_observation(game, previous, readers)
        except (RuntimeError, OSError):
            observation = None
        if observation is None:
            break
        observations.append(observation)
    passed = len(observations) == 2 and ready_pair(previous["score"], previous["kind"],
                                                    observations[0][0], observations[1][0])
    if passed and group == "early":
        early = observations[-1]
    else:
        wait_for_landing(game, previous, stop_file, False)
    return early, passed


def save_failure(previous, jump, landing, result, reason, score):
    path = ROOT / "last_jump_failure.zip"
    temp = path.with_suffix(".tmp")
    samples = [("takeoff.png", previous["frame"]), ("landing.png", landing)]
    samples.extend((f"flight_{i+1}_{offset:.3f}s.png", shot)
                   for i, (offset, shot) in enumerate(previous["flight"]))
    if result is not None:
        samples.append(("result.png", result))
    info = {"jump": jump, "reason": reason, "before": previous["score"], "observed_score": score,
            "foot": previous["foot"], "target": previous["target"], "length": previous["length"],
            "held": previous["duration"], "timing": previous["timing"], "model_version": previous.get("version"),
            "samples": [s[0] for s in samples]}
    with zipfile.ZipFile(temp, "w") as archive:
        for name, shot in samples:
            data = io.BytesIO()
            shot.save(data, format="PNG")
            archive.writestr(name, data.getvalue())
        archive.writestr("jump.json", json.dumps(info, ensure_ascii=False, indent=2))
    temp.replace(path)
    return str(path)


def measure_landing(game, frame, bbox, previous, score_reader, last_base, stop_file,
                    last_combo=None, last_combo_is_lower_bound=False):
    """Read one landing, attributing secondary HUD increases only once."""
    kind = previous["kind"]
    before = previous["score"]
    expected = {"record_player": 30, "rubiks_cube": 10}.get(kind)
    base_options = {1, min(32, last_base + 2)} if last_base is not None and last_base >= 2 else {1, 2}
    def plausible(n):
        return n is not None and (n in base_options if last_base is not None else n == 1 or (2 <= n <= 32 and n % 2 == 0))
    observations = []
    deadline = previous["release"] + 4.65
    credit = base = None
    source = "unresolved"
    while True:
        score = score_reader(frame)
        if score is not None and (before is None or score > before) and (not observations or score > observations[-1]):
            observations.append(score)
        total = observations[-1] - before if observations and before is not None else None
        if not kind:
            base = total if plausible(total) else None
            source = "ordinary"
            break
        if len(observations) >= 2:
            credit = observations[-1] - observations[0]
            base = observations[0] - before if before is not None else None
            source = "observed_secondary_increase"
            break
        if expected is not None and total is not None and not plausible(total) and plausible(total - expected):
            base = total - expected
            source = "combined_first_read"
            break
        if time.monotonic() >= deadline or stop_file.exists():
            if plausible(total):
                # With an unknown initial streak, an already-paid award can look like an even combo.
                ambiguous = last_base is None and (expected is None or plausible(total - expected))
                base = None if ambiguous else total
            source = "timeout" if not stop_file.exists() else "stopped"
            break
        for _ in range(3):
            if stop_file.exists():
                break
            time.sleep(.05)
        if not stop_file.exists():
            frame, bbox = game.frame()
    combo = lower_bound = capped = None
    if plausible(base):
        capped = base == 32
        if base == 1:
            combo, lower_bound = 0, False
        elif base == 2:
            combo, lower_bound = 1, False
        elif last_combo is not None:
            combo, lower_bound = last_combo + 1, last_combo_is_lower_bound
        else:
            combo, lower_bound = base // 2, True
    info = dict(kind=kind, before=before, readings=observations, score=score, base_points=base,
                observed_bonus=credit, inferred_bonus=expected if source == "combined_first_read" else None,
                attribution=source, combo=combo, combo_is_lower_bound=lower_bound, award_capped=capped)
    return frame, bbox, info


def play(args):
    output = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    def record(event, **values):
        row = {"time": datetime.now().isoformat(timespec="milliseconds"), "event": event, **values}
        line = json.dumps(row, ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
        print(line, flush=True)
        with (output / "events.jsonl").open("a", encoding="utf-8") as log:
            log.write(line + "\n")
    continuous = args.until_stopped
    stop_file = output / "stop"
    record("start", directory=str(output), jumps=None if continuous else args.jumps,
           restart=args.restart, until_stopped=continuous, stop_file=str(stop_file))
    training = OnlineTraining(stop_file, record)
    try:
        return play_rounds(args, output, stop_file, record, training)
    finally:
        training.close()


def play_rounds(args, output, stop_file, record, training):
    continuous = args.until_stopped
    current = ROOT / "current.png"
    if continuous:
        recent = ROOT / "recent"
        recent.mkdir(exist_ok=True)
        # The live frame plus the preceding 31 frames form one ring across process restarts.
        slots = [recent / f"frame_{i:02d}.png" for i in range(31)]
        slot_index = min(range(len(slots)), key=lambda i: slots[i].stat().st_mtime_ns if slots[i].exists() else -1)
    game = GameWindow(args.hwnd, background=continuous)
    if not continuous and game.user.GetForegroundWindow() != game.hwnd:
        print("请在 8 秒内将「跳一跳」游戏窗口切到前台。", file=sys.stderr, flush=True)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and game.user.GetForegroundWindow() != game.hwnd:
            time.sleep(.1)
    jumps = 0
    previous = None
    corrections = {}
    restarting = False
    readers = {}
    aim_counts = prior_aim_counts() if args.short_aim_test else {}
    marker_counts = prior_aim_counts("marker_stratum") if args.marker_aim_test else {}
    last_base = None
    last_combo, last_combo_is_lower_bound = None, False
    wait_candidates = 0
    round_jumps = 0
    while True:
        if stop_file.exists():
            record("stop", reason="stop_requested")
            return
        if continuous or jumps < args.jumps:
            training.tick()
        refresh_readers(readers, record)
        analyze = readers["jump_vision"][1].analyze
        score_module = readers["jump_score"][1]
        bonus_module = readers.get("jump_bonus", (None, None))[1]
        registration_module = readers.get("jump_registration", (None, None))[1]
        model_version, timing_model = readers.get("jump_timing", (None, None))
        visual_versions = tuple(readers.get(k, (None, None))[0] for k in ("jump_vision", "jump_score", "jump_bonus"))
        cached = previous.get("next_observation") if previous else None
        if cached and (previous["visual_versions"] != visual_versions or time.monotonic()-previous["release"] > 1.28):
            cached = None
            previous["next_wait"]["wait_applied"] = "normal"
            wait_for_landing(game, previous, stop_file, False)
        fall_frame = None
        # Allow a missed landing to finish its animation before considering another input.
        for attempt in range(3):
            reused = cached if attempt == 0 else None
            frame, bbox = reused[1] if reused else game.frame()
            if (attempt == 0 and previous and previous["input_wait"]["wait_applied"] == "early"
                    and previous["score"] is not None and score_module.restart_button(frame) is None
                    and (score_module.read_score(frame) or 0) <= previous["score"]):
                # Allow a late positive acknowledgement before treating an unchanged board as unaccepted.
                wait_for_landing(game, previous, stop_file, False,
                                 delay=max(1.55, time.monotonic()-previous["release"]+.25))
                if stop_file.exists():
                    record("stop", reason="stop_requested")
                    return
                reused = None
                frame, bbox = game.frame()
            initial_frame = frame
            if attempt == 0 and previous is not None and score_module.restart_button(frame) is None:
                frame, bbox, landing = measure_landing(game, frame, bbox, previous, score_module.read_score,
                                                       last_base, stop_file, last_combo, last_combo_is_lower_bound)
                record("landing", jump=jumps, timing=previous["timing"], model_version=previous["version"],
                       aim_group=previous["aim_group"], aim_stratum=previous["aim_stratum"],
                       aim_offset=previous["aim_offset"], length=round(previous["length"], 3),
                       **previous["input_wait"], **previous["marker_aim"], press_after_release=previous["interval"],
                       input_acknowledged=True if landing["readings"] and previous["score"] is not None else None,
                       **landing)
                last_base = landing["base_points"] if landing["combo"] is not None else None
                last_combo, last_combo_is_lower_bound = landing["combo"], landing["combo_is_lower_bound"]
            if stop_file.exists():
                record("stop", reason="stop_requested")
                return
            if reused and frame is initial_frame:
                _, score, scene = reused[0]
                button = None
            else:
                scene = analyze(frame)
                button = score_module.restart_button(frame)
                score = score_module.read_score(frame)
            if (previous and button is None and not scene.get("foot") and score is not None
                    and score == previous["score"] and falling_pawn(frame, previous["target"])):
                fall_frame = frame
                deadline = time.monotonic() + 5.
                while time.monotonic() < deadline and not stop_file.exists():
                    for _ in range(4):
                        if stop_file.exists():
                            break
                        time.sleep(.05)
                    if stop_file.exists():
                        break
                    frame, bbox = game.frame()
                    button = score_module.restart_button(frame)
                    score = score_module.read_score(frame)
                    if button is not None or (score is not None and score > previous["score"]):
                        break
                if stop_file.exists():
                    record("stop", reason="stop_requested")
                    return
                scene = analyze(frame)
                if button is None and (score is None or score <= previous["score"]):
                    path = save_failure(previous, jumps, fall_frame, None, "fall_without_result", score)
                    record("stop", reason="fall_without_result", score=score, diagnosis=path)
                    return
            target_kind = None
            if not reused and bonus_module and button is None and scene.get("foot"):
                locate = getattr(bonus_module, "locate_bonus_target", None)
                if locate is not None:
                    special = locate(frame, scene)
                    if special is not None:
                        target_kind, center = special
                        if target_kind == "rubiks_cube" or inspect_geometry(scene) is None:
                            scene = {**scene, "target": center, "confidence": max(.80, scene.get("confidence", 0.)),
                                     "reason": f"confirmed {target_kind} platform"}
                elif scene.get("target"):
                    target_kind = bonus_module.classify_target(frame, scene)
            geometry = inspect_geometry(scene)
            if button is not None or geometry is not None or attempt == 2:
                break
            time.sleep(.55)
        incoming_wait = previous["next_wait"]["wait_applied"] if previous else "normal"
        marker = marker_aim(scene, geometry, target_kind, incoming_wait, marker_counts,
                            args.marker_aim_test and button is None)
        if scene.get("marker_target") is not None:
            scene = {**scene, "marker_aim_plan": marker.get("marker_group", "geometry_unchanged"),
                     "aim_target": marker.get("marker_aim_target", scene.get("target"))}
        sample = f"{jumps:04d}_{'result' if button is not None else 'frame'}"
        if continuous:
            if current.exists():
                shutil.copyfile(current, slots[slot_index])
                slot_index = (slot_index + 1) % len(slots)
            new_frame = recent / "updating.tmp"
            frame.save(new_frame, format="PNG")
            new_frame.replace(current)
            image_path = str(current)
        else:
            frame.save(current)
            frame.save(output / (sample + ".png"))
            image_path = sample + ".png"
        record("observe", jump=jumps, score=score, button=button, scene=scene, image=image_path)
        if stop_file.exists():
            record("stop", reason="stop_requested")
            return
        if button is not None:
            if previous is not None:
                path = save_failure(previous, jumps, fall_frame if fall_frame is not None else frame,
                                    frame, "round_over", score)
                record("jump_failure", reason="round_over", jump=jumps, diagnosis=path,
                       **previous["input_wait"], press_after_release=previous["interval"])
            if not continuous:
                frame.save(ROOT / "last_result.png")
                if previous is not None:
                    previous["frame"].save(ROOT / "last_failed_jump.png")
            if not args.restart or (not continuous and jumps >= args.jumps) or restarting:
                record("stop", reason="round_over" if not restarting else "restart_not_acknowledged", score=score)
                return
            game.hold(bbox, button, .045)
            previous, restarting, last_base = None, True, None
            round_jumps = 0
            last_combo, last_combo_is_lower_bound = 0, False
            corrections.clear()
            record("restart", jump=jumps)
            time.sleep(1.65)
            continue
        restarting = False
        if (previous and previous["input_wait"]["wait_applied"] == "early" and
                (score is None or previous["score"] is None or score <= previous["score"])):
            same = score is not None and score == previous["score"] and scene.get("foot") and scene.get("target") and all(
                abs((scene["target"][i]-scene["foot"][i])-(previous["target"][i]-previous["foot"][i])) <= 2
                for i in (0,1))
            path = save_failure(previous, jumps, frame, None, "early_input_unverified", score)
            record("stop", reason="early_input_not_accepted" if same else "early_input_unverified",
                   input_acknowledged=False if same else None, diagnosis=path, **previous["input_wait"],
                   press_after_release=previous["interval"])
            return
        if previous is not None and scene.get("foot"):
            before = np.asarray(previous["frame"].convert("L"), dtype=np.int16)[390:790:3, ::3]
            after = np.asarray(frame.convert("L"), dtype=np.int16)[390:790:3, ::3]
            scene_difference = float(np.mean(np.abs(before - after)))
            if previous["score"] == score and scene_difference < .55:
                if score is not None:
                    path = save_failure(previous, jumps, frame, None, "jump_not_acknowledged", score)
                    record("jump_failure", jump=jumps, reason="jump_not_acknowledged", diagnosis=path)
                if not continuous:
                    frame.save(ROOT / "last_uncertain.png")
                record("stop", reason="jump_not_acknowledged", score=score)
                return
            fallback = getattr(registration_module, "fallback_shift", None)
            registration = registered_landing(previous, frame, scene, score, scene_difference > 1.5, fallback)
            if registration is not None:
                shift, actual, quality = registration
                if 25 <= actual <= 455 and abs(actual - previous["length"]) < 55:
                    applied_curve, residual_source = timing_prediction(actual, previous["model"])
                    residual = previous["duration"] - applied_curve
                    key = previous["timing_key"]
                    before_score = previous["score"]
                    gain = score - before_score if score is not None and before_score is not None else None
                    eligible = not quality["rejected"] and gain is not None and gain > 0 and previous["confidence"] >= .8
                    correction = corrections.get(key, 0.)
                    if eligible:
                        correction = float(np.clip(.70 * correction + .30 * residual, -.06, .06))
                        corrections[key] = correction
                    quality.update(takeoff_confidence=previous["confidence"], target_reason=previous["reason"],
                                   takeoff_score=before_score, landing_score=score, score_gain=gain,
                                   axis_error_px=round(actual-previous["length"], 3),
                                   aimed_axis_error_px=round(actual-previous["length"]-previous["aim_offset"]
                                                            -previous["marker_aim"].get("marker_axis_offset", 0.), 3))
                    record("calibration", jump=jumps, actual=round(actual, 1),
                           camera_shift=shift.round(1).tolist(), correction=round(correction, 4),
                           timing=previous["timing"], model_version=previous["version"],
                           residual_model_version=previous["version"] if residual_source == "model" else BASELINE_VERSION,
                           eligible=eligible, quality=quality, held=round(previous["duration"], 6),
                           aim_group=previous["aim_group"], aim_stratum=previous["aim_stratum"],
                           aim_offset=previous["aim_offset"],
                           **previous["marker_aim"],
                           length=round(previous["length"], 3), foot=previous["foot"],
                           target=previous["target"], landed_foot=scene["foot"], direction=previous["direction"])
                    if eligible:
                        training.pending += 1
        if geometry is None:
            if previous and scene.get("foot") and score is not None and score == previous["score"]:
                path = save_failure(previous, jumps, frame, None, "unchanged_score", score)
                record("jump_failure", jump=jumps, reason="unchanged_score", diagnosis=path)
            if not continuous:
                frame.save(ROOT / "last_uncertain.png")
            record("stop", reason="uncertain_geometry", scene=scene)
            return
        if not continuous and jumps >= args.jumps:
            record("stop", reason="batch_complete", score=score)
            return
        direction, length = geometry
        base_hold, timing = timing_prediction(length, timing_model)
        applied_model = timing_model if timing == "model" else None
        version = version_of(applied_model)
        key = (version, model_version if timing == "model" else None)
        aim_stratum, aim_group, aim_offset = short_aim(length, direction, aim_counts, args.short_aim_test)
        time_offset = aim_duration_delta(length, aim_offset, applied_model, timing, base_hold)
        marker_time = aim_duration_delta(length, marker.get("marker_axis_offset", 0.), applied_model, timing, base_hold)
        hold = max(.12, min(1.26, base_hold + time_offset + marker_time + corrections.get(key, 0.)))
        for xy in (scene["foot"], scene["target"]):
            game.point(bbox, xy)
        if stop_file.exists():
            record("stop", reason="stop_requested")
            return
        actual_time = game.hold(bbox, scene["foot"] if continuous else (268, 768), hold)
        released = time.monotonic()
        input_wait = previous["next_wait"] if previous is not None else dict(
            wait_group=None, wait_pair_ready=False, wait_applied="normal")
        interval = round(game.last_press-previous["release"], 4) if previous is not None else None
        jumps += 1
        round_jumps += 1
        if aim_stratum is not None:
            aim_counts[aim_stratum] = aim_counts.get(aim_stratum, 0) + 1
        if marker:
            mk = marker["marker_stratum"]
            marker_counts[mk] = marker_counts.get(mk, 0) + 1
        record("jump", jump=jumps, length=round(length, 1), direction=direction,
               requested=round(hold, 4), held=round(actual_time, 4), target_kind=target_kind,
               timing=timing, model_version=version, aim_group=aim_group, aim_stratum=aim_stratum,
               **input_wait, press_after_release=interval,
               **marker, marker_duration_delta=round(marker_time, 6),
               aim_offset=round(aim_offset, 4), aim_duration_delta=round(time_offset, 6),
               base_duration=round(base_hold, 4), correction=round(corrections.get(key, 0.), 4))
        previous = dict(frame=frame, score=score, foot=scene["foot"], direction=direction,
                        target=scene["target"], length=length, duration=actual_time, kind=target_kind,
                        release=released, timing=timing, timing_key=key, version=version, model=applied_model,
                        aim_group=aim_group, aim_stratum=aim_stratum, aim_offset=round(aim_offset, 4),
                        confidence=scene.get("confidence"), reason=scene.get("reason"),
                        input_wait=input_wait, interval=interval, visual_versions=visual_versions, marker_aim=marker)
        suspicious = attempt > 0 or scene.get("confidence", 0) < .90 or "confirmed " in scene.get("reason", "")
        previous["next_wait"] = dict(wait_group=None, wait_pair_ready=False, wait_applied="normal")
        if (continuous and round_jumps >= 2 and wait_candidates < 2*args.early_wait_pairs
                and not suspicious and scene["confidence"] >= .93 and target_kind is None and score is not None
                and 110 <= length <= 280):
            group = "control" if wait_candidates % 2 == 0 else "early"
            wait_candidates += 1
            early, passed = landing_trial(game, previous, stop_file, readers, group)
            previous["flight"] = []
            previous["next_observation"] = early
            previous["next_wait"] = dict(wait_group=group, wait_pair_ready=bool(passed),
                                         wait_applied="early" if early else "normal")
        else:
            previous["flight"] = wait_for_landing(game, previous, stop_file, length < 105 or suspicious)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--jumps", type=int, default=10, help="maximum jumps across all rounds")
    mode.add_argument("--until-stopped", action="store_true", help="keep playing until stopped or the screen is uncertain")
    parser.add_argument("--restart", action="store_true", help="restart only from a recognized result screen")
    parser.add_argument("--no-short-aim-test", dest="short_aim_test", action="store_false",
                        help="use the unmodified calibrated target at every distance")
    parser.add_argument("--early-wait-pairs", type=int, default=16,
                        help="number of alternating early/control pairs; 0 keeps the 1.30 s wait")
    parser.add_argument("--no-marker-aim-test", dest="marker_aim_test", action="store_false",
                        help="log measured markers but keep the geometric aim on all boards")
    parser.add_argument("--hwnd", type=lambda value: int(value, 0),
                        help="optional game window handle (decimal or 0x-prefixed hex)")
    args = parser.parse_args()
    if args.jumps < 1:
        parser.error("--jumps must be positive")
    if args.early_wait_pairs < 0:
        parser.error("--early-wait-pairs must be nonnegative")
    if sys.platform != "win32":
        parser.error("live game control requires Windows")
    try:
        play(args)
    except (OSError, RuntimeError, KeyboardInterrupt) as exc:
        print(json.dumps({"event":"stop", "reason":str(exc) or type(exc).__name__}, ensure_ascii=False), flush=True)
        raise SystemExit(2)
