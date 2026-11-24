"""Join frozen scenes to already-played game events for retrospective validation."""
from collections import defaultdict
import json
import numpy as np
from review_labels import ROOT,load_rows
logs=ROOT.parent.parent / 'runs'
index=defaultdict(list)
for log in sorted(logs.glob('*/events.jsonl')):
    previous=None
    motion=None
    pending={}
    for line in log.read_text(encoding='utf-8-sig').splitlines():
        try: e=json.loads(line)
        except ValueError:continue
        if e['event']=='restart':
            previous=None
            motion=None
        if e['event']=='observe':
            if motion and previous and e['jump']==motion['jump']:
                key=(previous.get('score'),tuple(previous.get('scene',{}).get('foot') or ()))
                record=pending.get(motion['jump'])
                if record:
                    record.update(after=e)
                    index[key].append(record)
            previous=e
            motion=None
        if e['event']=='jump':
            motion=e
            pending[e['jump']]={'run':log.parent.name,'hold':e,'before':previous}
        if e['event']=='landing' and e['jump'] in pending:
            pending[e['jump']]['landing']=e
        if e['event']=='calibration' and e['jump'] in pending:
            pending[e['jump']]['calibration']=e
byid={r['id']:r for r in load_rows()}
tests=json.loads((ROOT/'独立测试.json').read_text(encoding='utf-8'))['rows']
results=[]
for test in tests:
    row=byid[test['id']]
    attempts=index[(row['score'],tuple(row['foot']))]
    if len(attempts)!=1:
        continue
    attempt=attempts[0];after=attempt['after'];gain=after['score']-row['score'] if after['score'] is not None else None
    out=dict(id=row['id'],run=attempt['run'],jump=attempt['hold']['jump'],gain=gain,
             duration=attempt['hold']['held'],accepted=test['accepted'],visible=test['visible'],
             base_points=attempt.get('landing',{}).get('base_points'))
    cal=attempt.get('calibration',{})
    if cal.get('camera_shift') is not None and gain and gain>0 and cal.get('eligible',True) and after['scene'].get('foot'):
        new=np.array(after['scene']['foot'])-cal['camera_shift']
        real=np.array(test['ref']) if test['visible'] else None
        if real is not None:
            p=np.array(test['center'] if test['accepted'] else row['coarse'])
            old=np.array(row['coarse'])
            direction=attempt['hold']['direction']
            # Distances are observational because all recorded holds used the controller
            # deployed at the time, irrespective of this offline refinement.
            axis=lambda d:float((direction*d[0]-np.sqrt(3)*d[1])/2)
            out.update(live_landing_from_reference=new.tolist(),axis_from_coarse=round(axis(new-old),3),
                       axis_from_marker=round(axis(new-p),3))
    results.append(out)
summ=dict(heldout=len(tests),unique_action_pairs=len(results),positive_score=sum(r['gain'] is not None and r['gain']>0 for r in results),
          measured_landing_with_visible_center=sum('axis_from_marker' in r for r in results),
          center_credit=sum((r['base_points'] or 0)>=2 for r in results))
print(json.dumps(summ))
(ROOT/'落地反馈.json').write_text(json.dumps(dict(summary=summ,rows=results),ensure_ascii=False,indent=1),encoding='utf-8')
