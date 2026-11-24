from pathlib import Path
import time,json,math
import numpy as np
from sklearn.linear_model import LogisticRegression
from review_labels import ROOT,read_img,load_rows
from jump_refinement import MarkerRefiner,feature_map,extract_patch,RADIUS

def load_annotations():
    by_id={r['id']:r for r in json.loads((ROOT/'标注.json').read_text(encoding='utf-8'))}
    return [{**r,**by_id[r['id']]} for r in load_rows() if r['id'] in by_id and by_id[r['id']]['status'] in ('reviewed_marker','reviewed_no_marker')]
def fit(rows):
    rng=np.random.default_rng(83909)
    colors=[]
    for r in rows:
        if 'ref' not in r: continue
        x,y=np.rint(r['ref']).astype(int)-r['original_box'][:2]
        colors.append(np.median(np.asarray(read_img(r))[y-1:y+2,x-3:x+4],axis=(0,1)))
    prototype=np.median(colors,axis=0).tolist()
    xf=[];yf=[]
    grid_x,grid_y=np.meshgrid(np.arange(-RADIUS,RADIUS+1),np.arange(-RADIUS,RADIUS+1))
    for r in rows:
        image=read_img(r)
        patch=extract_patch(image,r['coarse'],r['original_box'][:2])
        if patch is None:continue
        features=feature_map(patch,prototype)
        if 'ref' in r:
            dx,dy=np.array(r['ref'])-r['coarse']
            radius=((grid_x-dx)/1.8)**2+((grid_y-dy)/1.35)**2
            positive=radius<=1
            hard=(radius>2)&(radius<22)
            other=radius>=22
        else:
            positive=np.zeros(grid_x.shape,dtype=bool)
            hard=np.zeros_like(positive)
            other=np.ones_like(positive)
        # Equal spatial sampling in every scene avoids domination by broad flat backgrounds.
        samples=rng.choice(np.flatnonzero(other),min(160,other.sum()),replace=False)
        idx=np.r_[np.flatnonzero(positive),np.flatnonzero(hard),samples]
        xf.append(features.reshape(-1,features.shape[-1])[idx])
        yf.append(positive.ravel()[idx])
    x,y=np.vstack(xf),np.concatenate(yf)
    center=x.mean(axis=0);scale=np.maximum(.035,x.std(axis=0))
    model=LogisticRegression(C=2.,class_weight='balanced',max_iter=800,random_state=100)
    model.fit((x-center)/scale,y)
    weights=model.coef_[0]/scale
    intercept=float(model.intercept_[0]-weights@center)
    return dict(prototype=prototype,weights=weights.tolist(),intercept=intercept,min_logit=-30.,
                min_gap=.65,max_correction_px=26.,train_scenes=len(rows),
                positive_patches=int(y.sum()),negative_patches=int((~y).sum()))
def evaluate(refiner,rows):
    result=[]
    for r in rows:
        image=read_img(r)
        t=time.perf_counter()
        p=refiner.predict(image,r['coarse'],r['original_box'][:2])
        ms=(time.perf_counter()-t)*1000
        ref=r.get('ref')
        error=lambda center:float(np.linalg.norm(np.array(center)-ref)) if ref else None
        dx,dy=np.array(r['coarse'])-r['foot'];axis=(abs(dx)-np.sqrt(3)*dy)/2
        result.append(dict(id=r['id'],batch=r['batch'],visible=ref is not None,accepted=p is not None,
                         short=bool(axis<115),start=r['coarse'],ref=ref,
                         center=None if p is None else p['center'],score=None if p is None else p['logit'],
                         old_error=error(r['coarse']),new_error=error(r['coarse'] if p is None else p['center']),ms=round(ms,3)))
    return result
def summaries(rows):
    markers=[r for r in rows if r['visible']]; accepted=[r for r in markers if r['accepted']]
    def stat(grp,k):
        a=[r[k] for r in grp]
        return None if not a else dict(n=len(a),mean=round(np.mean(a),3),p50=round(np.median(a),3),p95=round(np.percentile(a,95),3),max=round(max(a),3))
    return dict(n=len(rows),marks=len(markers),accepted_marks=len(accepted),
                false_positive=sum(r['accepted'] and not r['visible'] for r in rows),
                no_mark=len(rows)-len(markers),old_px=stat(markers,'old_error'),
                refined_px=stat(markers,'new_error'),end_to_end_ms=stat(rows,'ms'),
                old_short_px=stat([r for r in markers if r['short']],'old_error'),
                refined_short_px=stat([r for r in markers if r['short']],'new_error'))
def main():
    rows=load_annotations()
    train=[r for r in rows if r['batch'] not in ('B','C','D','E')];dev=[r for r in rows if r['batch']=='B']
    weights=fit(train)
    dst=ROOT/'marker_weights.json'
    dst.write_text(json.dumps(weights,indent=1),encoding='utf-8')
    model=MarkerRefiner(dst); development=evaluate(model,dev); training=evaluate(model,train)
    # Operate above every false alarm in development and training, with an extra logit margin.
    false_logits=[r['score'] for r in training+development if not r['visible'] and r['score'] is not None]
    weights['min_logit']=round(max(3., max(false_logits,default=-30.)+.7),3)
    dst.write_text(json.dumps(weights,indent=1),encoding='utf-8');model=MarkerRefiner(dst)
    results={name:evaluate(model,group) for name,group in [('train',train),('dev',dev)]}
    result={name:summaries(items) for name,items in results.items()};result['prototype']=weights['prototype'];result['threshold']=weights['min_logit']
    (ROOT/'开发评估.json').write_text(json.dumps(dict(summary=result,details=results),ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':
    main()
