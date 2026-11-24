"""Time-forward hold-duration calibration. No UI, vision or game-input operations."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
CUTOFF = "20261125_024210_579925"  # Last complete segment before the requested restart.
POINTS = np.array([[65,230],[102,337],[125,418.9],[170,550.6],[220,710.7],[268,842.9]])
NAMES = ("fixed", "linear", "piecewise", "mlp")
INITIAL_SEGMENTS = 3
SEED = 764

def samples():
    groups = []
    for path in sorted((ROOT/"runs").glob("*/events.jsonl")):
        if path.parent.name > CUTOFF:
            continue
        held, group = {}, []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            event = json.loads(line)
            if event["event"] == "jump":
                held[event["jump"]] = float(event["held"])*1000
            if event["event"] == "calibration" and event["jump"] in held:
                group.append([float(event["actual"]),held.pop(event["jump"])])
        if group:
            groups.append((path.parent.name,np.array(group)))
    return groups

def fixed(x):
    x=np.asarray(x)
    yp=np.interp(x, POINTS[:,0], POINTS[:,1])
    return np.where(x<65,230+(x-65)*107/37,np.where(x>268,842.9+(x-268)*1000/485,yp))

def features(x,piecewise):
    col=[np.ones(len(x)),(x-165)/100]
    if piecewise:
        col.extend([np.maximum(0,(x-115)/100),np.maximum(0,(x-200)/100)])
    return np.stack(col,axis=1)

def robust_fit(x,y,piecewise=False):
    a=features(x,piecewise)
    def residuals(w):
        errors=a@w-y
        return np.r_[errors,0.9*w[2:]]  # Mild shrinkage only on hinge slope changes.
    return least_squares(residuals,np.linalg.lstsq(a,y,rcond=None)[0],loss="huber",
                         f_scale=7.,max_nfev=200).x

def train_mlp(x,y,linear):
    # The linear physical response is a skip connection. Learn only the systematic
    # departures; scale a 10 ms residual to one output unit. Two trainable hidden layers.
    linear_prediction=features(x,False)@linear
    xx=((x-165)/100)[:,None]
    yy=((y-linear_prediction)/10)[:,None]
    rng=np.random.default_rng(SEED)
    w=[rng.normal(0,0.5,(1,6)),np.zeros(6),rng.normal(0,0.32,(6,4)),
       np.zeros(4),rng.normal(0,0.10,(4,1)),np.zeros(1)]
    m=[np.zeros_like(t) for t in w]; v=[np.zeros_like(t) for t in w]
    for step in range(1,901):
        h1=np.tanh(xx@w[0]+w[1]); h2=np.tanh(h1@w[2]+w[3])
        out=h2@w[4]+w[5]
        grad=np.clip(out-yy,-0.7,0.7)/len(x)
        d2=(grad@w[4].T)*(1-h2*h2)
        d1=(d2@w[2].T)*(1-h1*h1)
        gs=[xx.T@d1,d1.sum(axis=0),h1.T@d2,d2.sum(axis=0),h2.T@grad,grad.sum(axis=0)]
        for k in range(len(w)):
            g=gs[k]+(0.003*w[k] if k%2==0 else 0)
            m[k]=0.9*m[k]+0.1*g; v[k]=0.999*v[k]+0.001*g*g
            w[k]-=.006*(m[k]/(1-0.9**step))/(np.sqrt(v[k]/(1-0.999**step))+1e-8)
    return w

def predict_mlp(x,linear,params):
    w,b,u,c,v,d=params
    norm=((x-165)/100)[:,None]
    return features(x,False)@linear+10*(np.tanh(np.tanh(norm@w+b)@u+c)@v+d).ravel()

def fit_models(train):
    x,y=train.T
    lin=robust_fit(x,y)
    return lin,robust_fit(x,y,True),train_mlp(x,y,lin)

def predictions(train,future):
    lin,seg,weights=fit_models(train)
    x=future[:,0]
    return np.stack([fixed(x),features(x,False)@lin,features(x,True)@seg,predict_mlp(x,lin,weights)],axis=1)

def metrics(rows):
    result={}
    for key,cond in (("all",np.ones(len(rows),dtype=bool)),("short",rows[:,0]<115)):
        a=rows[cond]; result[key]={"n":len(a)}
        for i,name in enumerate(NAMES):
            if not len(a):continue
            e=a[:,i+2]-a[:,1]
            result[key][name]={"mae_ms":round(float(np.abs(e).mean()),3),
                               "bias_ms":round(float(e.mean()),3),
                               "rmse_ms":round(float(np.sqrt((e*e).mean())),3),
                               "p90_abs_ms":round(float(np.percentile(np.abs(e),90)),3)}
    return result

def main():
    groups=samples()
    assert groups[-1][0]==CUTOFF
    folds=[]; raw=[]
    for k in range(INITIAL_SEGMENTS,len(groups)):
        name,future=groups[k]
        train=np.vstack([g for _,g in groups[:k]])
        out=np.column_stack((future,predictions(train,future)))
        folds.append({"segment":name,"train":len(train),"metrics":metrics(out)})
        raw.append(out)
    everything=np.vstack([g for _,g in groups])
    train=np.vstack([g for _,g in groups[:-1]])
    chronological=np.vstack(raw[:-1])
    result={"cutoff":CUTOFF,"source_rows":len(everything),"source_segments":len(groups),
            "range":everything[:,0].min().item(),"max":everything[:,0].max().item(),
            "training_range":train[:,0].min().item(),"training_max":train[:,0].max().item(),
            "source_short":int(np.sum(everything[:,0]<115)),"folds":folds,
            "rolling":metrics(chronological),"latest":metrics(raw[-1]),
            "latest_rows":np.round(raw[-1],3).tolist()}
    print(json.dumps(result,indent=1))
    (ROOT/"duration_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    # Retain offline parameters trained BEFORE independent test. No automatic activation.
    lin,seg,weights=fit_models(train)
    obj={"architecture":[1,6,4,1],"trained_through":groups[-2][0],
         "linear_ms":lin.tolist(),"hidden_weights":[w.tolist() for w in weights],
         "applicable_px":[float(train[:,0].min()),float(train[:,0].max())]}
    (ROOT/"duration_mlp_offline.json").write_text(json.dumps(obj),encoding="utf-8")

if __name__=="__main__":
    main()
