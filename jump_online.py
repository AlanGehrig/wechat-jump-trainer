"""Incrementally collect real jump measurements and promote prospectively tested timing models."""
import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from train_duration import fit_models, fixed as historical_fixed

ROOT=Path(__file__).resolve().parent
DEFAULT_STORE=ROOT / "跳一跳训练"
BATCH=12
ORIGINAL_CUTOFF="20261125_024210_579925"
METADATA_MARK="# Online training metadata."
FORMULA_VERSION="fixed-tail-320-side12-and-score-v2"
def fixed(x):
    x=np.asarray(x)
    return np.where(x>268,842.9+(x-268)*1000/320,historical_fixed(x))
def now():
    return datetime.now().isoformat(timespec="milliseconds")
def compact(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)
def fingerprint(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
def write_atomic(path,text):
    tmp=None
    try:
        with tempfile.NamedTemporaryFile("w",dir=path.parent,prefix=path.name+".",
                                         suffix=".tmp",encoding="utf-8",delete=False) as stream:
            tmp=Path(stream.name);stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if tmp and tmp.exists():tmp.unlink()

@contextmanager
def single_writer(path):
    import msvcrt
    with open(path,"a+b") as guard:
        guard.seek(0,2)
        if guard.tell()==0:guard.write(b"0");guard.flush()
        guard.seek(0)
        try:msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:yield False;return
        try:yield True
        finally:
            guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_UNLCK,1)

def schema(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS cursors(run TEXT PRIMARY KEY,offset INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS observations(run TEXT,jump INTEGER,at TEXT,score INTEGER,origin TEXT,
        foot TEXT,target TEXT,confidence REAL,PRIMARY KEY(run,jump,at));
    CREATE TABLE IF NOT EXISTS label_review(run TEXT,jump INTEGER,status TEXT,gain INTEGER,
        lateral_px REAL,score_provenance TEXT,PRIMARY KEY(run,jump));
    CREATE TABLE IF NOT EXISTS holds(run TEXT,jump INTEGER,release TEXT,held REAL,
        requested REAL,length REAL,direction INTEGER,version TEXT,PRIMARY KEY(run,jump));
    CREATE TABLE IF NOT EXISTS measurements(run TEXT,jump INTEGER,measured TEXT,actual REAL,
        shift TEXT,quality TEXT,eligible INTEGER,PRIMARY KEY(run,jump));
    CREATE TABLE IF NOT EXISTS rounds(run TEXT,jump INTEGER,at TEXT,PRIMARY KEY(run,jump));
    CREATE TABLE IF NOT EXISTS failures(run TEXT,at TEXT,jump INTEGER,status TEXT,reason TEXT,
        PRIMARY KEY(run,at,status));
    CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY,created TEXT,ready TEXT,train_n INTEGER,
        train_end TEXT,source TEXT,incumbent TEXT,incumbent_sha TEXT,state TEXT,eval_json TEXT);
    CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,created TEXT,status TEXT,metrics TEXT);
    CREATE INDEX IF NOT EXISTS measured_time ON measurements(measured);
    """)
    if db.execute("PRAGMA user_version").fetchone()[0]<3:
        db.execute("DELETE FROM cursors")
        db.execute("PRAGMA user_version=3")
        db.commit()

def observation_score(e,run_dir):
    v=e.get("score")
    if isinstance(v,int) and v>=0:return v,"game_hud"
    name=e.get("image","")
    filename=Path(name)
    if (filename.is_absolute() or filename.name!=name or
            not re.fullmatch(r"[0-9]+_frame[.]png",name or "")):return None,"unknown"
    frozen=run_dir/filename
    if not frozen.is_file():return None,"unknown"
    from PIL import Image
    from jump_score import read_score
    with Image.open(frozen) as frame:v=read_score(frame)
    return (v,"stored_frame_ocr") if v is not None else (None,"unknown")

def ingest(db,runs):
    added=0
    for p in sorted(runs.glob("*/events.jsonl")):
        run=p.parent.name
        row=db.execute("SELECT offset FROM cursors WHERE run=?",(run,)).fetchone()
        offset=row[0] if row else 0
        if p.stat().st_size<offset:offset=0
        with p.open("rb") as f:
            f.seek(offset)
            while True:
                start=f.tell();line=f.readline()
                if not line.endswith(b"\n"):offset=start;break
                e=json.loads(line.decode("utf-8-sig"));k=e.get("event");j=e.get("jump");t=e.get("time")
                if k=="jump" and j is not None:
                    db.execute("""INSERT INTO holds VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(run,jump)
                               DO UPDATE SET version=COALESCE(holds.version,excluded.version)""",
                        (run,j,t,e.get("held"),e.get("requested"),e.get("length"),e.get("direction"),e.get("model_version") or e.get("timing_version")))
                elif k=="calibration" and j is not None and e.get("actual") is not None:
                    v=float(e["actual"]);quality=e.get("quality")
                    ok=e.get("eligible",True) is not False and math.isfinite(v) and 25<=v<=355
                    q=db.execute("INSERT OR IGNORE INTO measurements VALUES(?,?,?,?,?,?,?)",
                        (run,j,t,v,compact(e.get("camera_shift")),compact(quality),int(ok)))
                    added+=int(ok)*q.rowcount
                elif k=="observe" and j is not None:
                    scene=e.get("scene") or {};score,origin=observation_score(e,p.parent)
                    db.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?)",
                               (run,j,t,score,origin,compact(scene.get("foot")),compact(scene.get("target")),scene.get("confidence")))
                    if e.get("button") is not None:
                        db.execute("INSERT OR IGNORE INTO failures VALUES(?,?,?,?,?)",(run,t,j,"result_screen","replay_button"))
                elif k=="jump_failure" and j is not None:
                    reason=e.get("reason")
                    state="confirmed_fall" if reason=="round_over" else "failed_or_unverified"
                    db.execute("INSERT OR IGNORE INTO failures VALUES(?,?,?,?,?)",(run,t,j,state,reason))
                elif k=="restart" and j is not None:
                    db.execute("INSERT OR IGNORE INTO rounds VALUES(?,?,?)",(run,j,t))
                elif k=="stop" and e.get("reason") in ("round_over","fall_without_result"):
                    db.execute("INSERT OR IGNORE INTO failures VALUES(?,?,?,?,?)",
                               (run,t,j,"fall_suspected" if e["reason"]=="fall_without_result" else "result_screen",e["reason"]))
            db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?)",(run,offset))
    db.commit()
    return added

def refresh_review(db):
    rows=db.execute("""SELECT h.run,h.jump,h.release,h.direction,h.length,m.measured,m.actual,
                      m.shift,m.eligible,m.quality FROM holds h JOIN measurements m USING(run,jump)
                      LEFT JOIN label_review v USING(run,jump) WHERE v.status IS NULL""").fetchall()
    for run,j,release,direction,length,measured,actual,shift,eligible,quality in rows:
        before=db.execute("""SELECT score,origin,foot,target,confidence FROM observations
                       WHERE run=? AND jump=? AND at<=? ORDER BY at DESC LIMIT 1""",(run,j-1,release)).fetchone()
        after=db.execute("""SELECT score,origin,foot,target,confidence FROM observations
                      WHERE run=? AND jump=? AND at>=? AND at<=? ORDER BY at DESC LIMIT 1""",(run,j,release,measured)).fetchone()
        q=json.loads(quality) or {};delta=None;side=None;provenance=None;status="eligible"
        confirmed=db.execute("""SELECT 1 FROM failures WHERE run=? AND jump=?
                      AND status IN ('confirmed_fall','failed_or_unverified')""",(run,j)).fetchone()
        if not eligible or confirmed:status="invalid_calibration_or_failed"
        elif not(before and after):status="missing_landing_observation"
        else:
            b,a=before[0],after[0]
            delta=None if b is None or a is None else a-b
            provenance=before[1]+"/"+after[1]
            foot,target,landed=map(json.loads,(before[2],before[3],after[2]))
            move=json.loads(shift)
            if delta is None:status="score_unknown"
            elif delta<=0:status="score_did_not_increase"
            elif not (foot and target and landed and move and direction in (-1,1)):status="geometry_unknown"
            elif (before[4] or 0)<.8 or not (25<=actual<=355) or abs(actual-length)>=55:status="low_confidence_geometry"
            else:
                dx=landed[0]-move[0]-target[0];dy=landed[1]-move[1]-target[1]
                side=abs((direction*dx+math.sqrt(3)*dy)/2)
                if side>12:status="off_axis_landing"
                if q and any(k in q for k in ("inliers","inlier_ratio","feature_span_px")):
                    if (q.get("inliers",0)<7 or q.get("inlier_ratio",0)<.48 or
                        q.get("feature_span_px",0)<58 or q.get("median_inlier_residual_px",999)>3.0):
                        status="weak_orb_registration"
        db.execute("INSERT OR REPLACE INTO label_review VALUES(?,?,?,?,?,?)",(run,j,status,delta,side,provenance))
    db.commit()
def dataset(db):
    return db.execute("""SELECT h.run,h.jump,h.release,h.held,m.measured,m.actual,
              COALESCE((SELECT MAX(r.at) FROM rounds r WHERE r.run=h.run AND r.jump<h.jump),'initial')
              FROM measurements m JOIN holds h USING(run,jump) JOIN label_review v USING(run,jump)
              WHERE v.status='eligible' AND m.eligible=1 AND h.held>0.05 AND h.held<2.5 AND h.release<=m.measured
              ORDER BY m.measured,h.run,h.jump""").fetchall()

def module(source):
    ns={};exec(compile(source,"jump_timing.py","exec"),ns)
    return ns

def make_source(incumbent,lin,weights,version,metadata):
    # Preserve the incumbent's proven feature scaling, fallback and blending contract.
    tail=incumbent.split("POINTS = ",1)[1].split(METADATA_MARK,1)[0].rstrip()
    tail,count=re.subn(r"(POINTS\[-1\]\[1\] \+ \(distance-POINTS\[-1\]\[0\]\)/)(?:485|320)",r"\g<1>320",tail)
    if count!=1:raise RuntimeError("Cannot verify candidate fallback tail")
    model=('"""Live-trained residual MLP for WeChat Jump Jump. Returns seconds or None."""\n'
           'from math import isfinite, tanh\n'
           +'LINEAR = '+repr(lin.tolist())+'\nWEIGHTS = '+repr([w.tolist() for w in weights])
           +'\nPOINTS = '+tail+'\n\n'+METADATA_MARK+'\nMODEL_VERSION = '+repr(version)
           +'\nTRAINING_META = '+repr(metadata)+'\n')
    g=module(model)["predict_hold"]
    grid=np.linspace(85,285,4001);p=np.array([g(d) for d in grid])
    if (not np.isfinite(p).all() or not np.all(np.diff(p)>0)
            or g(84.9) is not None or g(285.1) is not None
            or not np.allclose(p[[0,-1]],fixed(np.array([85,285]))/1000,atol=1e-9)):
        raise ValueError("Candidate failed range, monotonicity or edge continuity checks")
    return model

def create_candidate(db,rows,incumbent):
    pre=now(); ver="online-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    train=np.array([(r[5],1000*r[3]) for r in rows])
    lin,_,weights=fit_models(train)
    meta={"trained_samples":len(rows),"train_measured_through":rows[-1][4],
          "fallback_version":FORMULA_VERSION,"training_run":rows[-1][0],"training_jump":rows[-1][1],"train_started":pre,
          "features":"camera-corrected axial pixels, 543px reference","domain_px":[85,285]}
    source=make_source(incumbent,lin,weights,ver,meta)
    ready=now()
    db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ver,pre,ready,len(rows),rows[-1][4],source,incumbent,fingerprint(incumbent),"pending",None))
    db.commit()
    return {"version":ver,"training_samples":len(rows),"ready":ready}

def evaluate(source,incumbent,heldout):
    x=np.array([r[5] for r in heldout]); y=np.array([r[3]*1000 for r in heldout])
    curves={"candidate":module(source)["predict_hold"],"incumbent":module(incumbent)["predict_hold"]}
    pred={"fixed":fixed(x)}
    for name,func in curves.items():
        vals=[func(v) for v in x]
        pred[name]=np.array([base if val is None else 1000*val for val,base in zip(vals,pred["fixed"])])
    metrics={}
    for name,mask in (("all",np.full(len(x),True)),("short",x<115)):
        m={"n":int(mask.sum())}
        if mask.any():
            for method,est in pred.items():
                e=est[mask]-y[mask]
                m[method]={"mae_ms":round(float(np.abs(e).mean()),4),
                           "rmse_ms":round(float(np.sqrt((e**2).mean())),4),
                           "p90_ms":round(float(np.percentile(np.abs(e),90)),4)}
        metrics[name]=m
    return metrics

def gate(metrics):
    m=metrics["all"];c=m["candidate"];a=m["incumbent"];b=m["fixed"]
    shortlist=metrics["short"]
    rules={"beats_incumbent":c["mae_ms"]+0.12<=a["mae_ms"],
           "beats_fixed":c["mae_ms"]<=b["mae_ms"],
           "no_large_tail_regression":c["p90_ms"]<=a["p90_ms"]+1,
           "short_non_regression":shortlist["n"]<3 or shortlist["candidate"]["mae_ms"]<=shortlist["incumbent"]["mae_ms"]+0.5}
    return all(rules.values()),rules

def publish_metrics(db,store):
    rows=db.execute("SELECT id,created,status,metrics FROM batches ORDER BY created").fetchall()
    write_atomic(store/"批次评估.jsonl",''.join(compact({"id":i,"time":t,"status":s,**json.loads(m)})+"\n" for i,t,s,m in rows))

def step(args):
    store=args.store;store.mkdir(parents=True,exist_ok=True)
    with single_writer(store/"trainer.lock") as owns:
        if not owns:return {"status":"busy"}
        with closing(sqlite3.connect(store/"跳跃数据.sqlite3",timeout=4)) as db, db:
            schema(db);added=ingest(db,args.runs);refresh_review(db);rows=dataset(db)
            # If the process crashed after publishing, atomically replaced model metadata
            # is authoritative. Reconcile the DB before processing the next batch.
            incumbent=args.model.read_text(encoding="utf-8-sig")
            live=module(incumbent);version=live.get("MODEL_VERSION","initial97-"+fingerprint(incumbent)[:10])
            old=db.execute("SELECT state,eval_json,source FROM candidates WHERE id=?",(version,)).fetchone()
            if old and old[0]=="qualified" and old[1] and not args.preview:
                write_atomic(store/"最新部署模型.py",old[2])
                db.execute("UPDATE candidates SET state='promoted' WHERE id=?",(version,))
                db.execute("UPDATE batches SET status='promoted' WHERE id=?",(version,));db.commit()
            if not (store/"最新部署模型.py").exists() and not args.preview:
                write_atomic(store/"最新部署模型.py",incumbent)
            mtime=args.model.stat().st_mtime_ns
            common={"labels":len(rows),"inserted_calibrations":added,
                    "quality":dict(db.execute("SELECT status,count(*) FROM label_review GROUP BY status").fetchall()),
                    "actions":db.execute("SELECT count(*) FROM holds").fetchone()[0],
                    "unpaired_actions":db.execute("""SELECT count(*) FROM holds h WHERE NOT EXISTS
                        (SELECT 1 FROM measurements m WHERE m.run=h.run AND m.jump=h.jump AND m.eligible=1)""").fetchone()[0],
                    "confirmed_result_frames":db.execute("SELECT count(*) FROM failures WHERE status='result_screen'").fetchone()[0],
                    "unverified_fall_signals":db.execute("SELECT count(*) FROM failures WHERE status='fall_suspected'").fetchone()[0],
                    "active":version}
            if not rows:return {"status":"no_measurements",**common}
            pending=db.execute("""SELECT id,ready,train_n,source,incumbent,incumbent_sha FROM candidates
                WHERE state IN ('pending','qualified') ORDER BY ready DESC LIMIT 1""").fetchone()
            if pending and (fingerprint(incumbent)!=pending[5] or
                            module(pending[3]).get("TRAINING_META",{}).get("fallback_version")!=FORMULA_VERSION):
                db.execute("UPDATE candidates SET state='superseded' WHERE id=?",(pending[0],));db.commit();pending=None
            if not pending:
                if len(rows)<24:return {"status":"collecting",**common}
                out=create_candidate(db,rows,incumbent);publish_metrics(db,store)
                return {"status":"trained_awaiting_future","pending":out,"required":BATCH,**common}
            ident,ready,ntrain,source,original,sha=pending
            # The complete input gesture must start AFTER model fitting has finished.
            # Distinct (run, restart epoch) groups cannot be mixed into a validation set.
            ready_dt=datetime.fromisoformat(ready);groups=defaultdict(list)
            for r in rows:
                if r[4]<=ready:continue
                started=datetime.fromisoformat(r[2])-timedelta(seconds=r[3])
                if started>=ready_dt:
                    groups[(r[0],r[6])].append(r)
            enough=[v[:BATCH] for v in groups.values() if len(v)>=BATCH]
            if not enough:
                return {"status":"awaiting_future","candidate":ident,"required":BATCH,
                        "largest_single_round_batch":max(map(len,groups.values()),default=0),**common}
            selected=min(enough,key=lambda batch:batch[-1][4])
            met=evaluate(source,original,selected);passed,rules=gate(met)
            detail={"candidate":ident,"incumbent":version,"train_n":ntrain,"ready":ready,
                    "test_n":len(selected),"test_run":selected[0][0],"round_boundary":selected[0][6],
                    "test_jumps":[r[1] for r in selected],"test_first_release":selected[0][2],
                    "test_last_measurement":selected[-1][4],"metrics":met,"gates":rules}
            state="qualified" if passed else "rejected"
            db.execute("UPDATE candidates SET state=?,eval_json=? WHERE id=?",(state,compact(detail),ident))
            db.execute("INSERT OR REPLACE INTO batches VALUES(?,?,?,?)",(ident,now(),state,compact(detail)));db.commit()
            if passed and args.preview:
                publish_metrics(db,store);return {"status":"qualified_preview","evaluation":detail,**common}
            if passed:
                if (args.model.stat().st_mtime_ns!=mtime or
                    fingerprint(args.model.read_text(encoding="utf-8-sig"))!=sha):
                    raise RuntimeError("Active model changed during validation")
                write_atomic(args.model,source);write_atomic(store/"最新部署模型.py",source)
                db.execute("UPDATE candidates SET state='promoted' WHERE id=?",(ident,))
                db.execute("UPDATE batches SET status='promoted' WHERE id=?",(ident,));db.commit()
                common["active"]=ident;incumbent=source
            # The next candidate may now train on earlier validation observations.
            nxt=create_candidate(db,rows,incumbent)
            publish_metrics(db,store)
            return {"status":"promoted" if passed else "incumbent_retained","evaluation":detail,
                    "pending":nxt,**common}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--once",action="store_true",help="Collect, assess and train once, then exit.")
    p.add_argument("--watch",action="store_true",help="Repeat until stopped externally.")
    p.add_argument("--interval",type=float,default=30)
    p.add_argument("--store",type=Path,default=DEFAULT_STORE)
    p.add_argument("--runs",type=Path,default=ROOT/"runs")
    p.add_argument("--model",type=Path,default=ROOT/"jump_timing.py")
    p.add_argument("--preview",action="store_true",help="Collect and fit but do not publish the active model.")
    a=p.parse_args()
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):stream.reconfigure(encoding="utf-8")
    while True:
        print(compact(step(a)),flush=True)
        if not a.watch or a.once:return
        time.sleep(max(1,a.interval))

if __name__=="__main__":
    main()
