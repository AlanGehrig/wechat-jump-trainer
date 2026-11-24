"""Freeze existing game images, keeping only the central playfield for offline experiments."""
from pathlib import Path
from PIL import Image
import argparse
import hashlib
import json
import sys
import io
from datetime import datetime

DEST = Path(__file__).resolve().parent
SOURCE = DEST.parent.parent
sys.path.insert(0, str(SOURCE))
from jump_score import read_score, restart_button
from jump_vision import analyze

def capture(batch, include_old=False):
    output = DEST / "画面"
    output.mkdir(exist_ok=True)
    catalog = DEST / "样本.jsonl"
    rows = [json.loads(i) for i in catalog.read_text(encoding="utf-8").splitlines()] if catalog.exists() else []
    seen = {r["digest"] for r in rows}
    samples = []
    if include_old:
        for run in sorted((SOURCE / "runs").glob("*")):
            samples.extend(("历史", p) for p in sorted(run.glob("*_frame.png")))
        samples.extend(("历史", p) for p in sorted(SOURCE.glob("sample*.png")))
        samples.append(("历史",SOURCE/"frame.png"))
    recent = list((SOURCE/"recent").glob("frame_*.png"))
    if (SOURCE/"current.png").is_file():
        recent.append(SOURCE/"current.png")
    recent.sort(key=lambda p: p.stat().st_mtime_ns)
    samples.extend((batch,p) for p in recent)
    added = []
    for group,path in samples:
        try:
            buf = path.read_bytes()
            frame = Image.open(io.BytesIO(buf)).convert("RGB")
            if frame.size != (543,1029) or restart_button(frame):
                continue
            score = read_score(frame)
            scene = analyze(frame)
            if score is None or not scene["foot"] or not scene["target"]:
                continue
            crop = frame.crop((4,335,539,730))
            digest = hashlib.sha256(crop.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            name = f"v{len(rows)+len(added)+1:04d}"
            crop.save(output/(name+".png"))
            added.append(dict(id=name,batch=group,source=path.name,score=score,
                              digest=digest,foot=scene["foot"],coarse=scene["target"],reason=scene["reason"],
                              original_box=[4,335,539,730],captured=datetime.now().isoformat(timespec='seconds')))
        except OSError:
            continue
    with catalog.open("a",encoding="utf-8") as log:
        for row in added:
            log.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("saved",len(added),"total",len(rows)+len(added))
    from collections import Counter
    print(Counter(r['batch'] for r in rows+added))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", help="label for newly collected playfield frames")
    parser.add_argument("--early", action="store_true", help="also collect older full game frames")
    args = parser.parse_args()
    capture(args.batch, args.early)
