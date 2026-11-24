"""Lightweight center-marker localization. This module never sends game input."""
from pathlib import Path
import json
import cv2
import numpy as np

# Input points use the 543x1029 game window. Full window arrays and cropped
# playfields with a known origin use the same local search and fitted weights.
RADIUS = 28
PAD = 15
_X, _Y = np.meshgrid(np.arange(-13,14),np.arange(-9,10))
def kernel(mask):
    return np.asarray(mask,dtype=np.float32) / np.count_nonzero(mask)
ellipse = (_X/9.1)**2 + (_Y/5.25)**2
inner = kernel(ellipse <= .55)
disk = kernel(ellipse <= 1.)
outer_band = (ellipse > 1.45) & (ellipse <= 2.10)
outer = kernel(outer_band)
sides = [kernel(outer_band & condition) for condition in (_X < -3, _X > 3, _Y < -2, _Y > 2)]


def feature_map(patch, prototype):
    rgb = patch.astype(np.float32)
    shade = np.mean(rgb,axis=2)
    deviation = np.max(np.abs(rgb - np.float32(prototype)),axis=2)
    # Photometric bandwidths keep both high-contrast coloured faces and
    # white-on-white markers. Each fitted coefficient is learned on local frames.
    features = []
    for bandwidth in (.55,2.,8.,22.):
        similarity = np.exp(-.5*(deviation/bandwidth)**2)
        core = cv2.filter2D(similarity,-1,inner)
        face = cv2.filter2D(similarity,-1,disk)
        rim = cv2.filter2D(similarity,-1,outer)
        directional = [cv2.filter2D(similarity,-1,k) for k in sides]
        features.extend((core,face,face-rim,face-np.maximum.reduce(directional),face-np.minimum.reduce(directional)))
    core = cv2.filter2D(shade,-1,inner)
    rim = cv2.filter2D(shade,-1,outer)
    variance = np.maximum(0,cv2.filter2D(shade*shade,-1,inner)-core*core)
    features.extend((core/255.,np.abs(core-rim)/80.,np.sqrt(variance)/25.))
    return np.stack(features,axis=-1)[PAD:-PAD,PAD:-PAD]


def extract_patch(image, point, origin=(0,0)):
    x,y = np.rint(np.asarray(point)-origin).astype(int)
    r = RADIUS + PAD
    if hasattr(image, "crop"):
        if x-r<0 or y-r<0 or x+r>=image.width or y+r>=image.height:
            return None
        return np.asarray(image.crop((x-r,y-r,x+r+1,y+r+1)).convert('RGB'))
    if x-r<0 or y-r<0 or x+r>=image.shape[1] or y+r>=image.shape[0]:
        return None
    return image[y-r:y+r+1,x-r:x+r+1,:3]


class MarkerRefiner:
    def __init__(self, path=None):
        path = Path(path) if path is not None else Path(__file__).with_name('marker_weights.json')
        self.params=json.loads(path.read_text(encoding='utf-8'))
        self.proto=np.rint(self.params['prototype']).astype(np.int16)
        self.weights=np.array(self.params['weights'],dtype=np.float32)

    def predict(self,image,point,origin=(0,0)):
        patch=extract_patch(image,point,origin)
        if patch is None:
            return None
        # Use broad colour and size proposals solely to limit computation.
        # The fitted model decides whether a proposal resembles a real marker
        # and localizes its center from the original RGB neighbourhood.
        half=RADIUS+PAD
        tol=np.max(np.abs(patch.astype(np.int16)-self.proto),axis=2)
        proposals=[]
        for level in (0,3):
            _,_,regions,centers=cv2.connectedComponentsWithStats((tol<=level).astype(np.uint8))
            for (left,top,width,height,area),center in zip(regions[1:],centers[1:]):
                if not (8<=width<=30 and 4<=height<=20 and 24<=area<=425):
                    continue
                x,y=np.rint(center).astype(int)
                if max(abs(x-half),abs(y-half))>RADIUS or any(abs(x-u)<4 and abs(y-v)<3 for u,v in proposals):
                    continue
                proposals.append((x,y))
        scored=[]
        reach=PAD+4
        for x0,y0 in proposals:
            if (x0-reach<0 or y0-reach<0 or x0+reach>=patch.shape[1] or y0+reach>=patch.shape[0]):
                continue
            tile=patch[y0-reach:y0+reach+1,x0-reach:x0+reach+1]
            values=feature_map(tile,self.params['prototype'])@self.weights+self.params['intercept']
            y,x=np.unravel_index(np.argmax(values),values.shape)
            peak=float(values[y,x])
            if peak<self.params['min_logit'] or not (1<=x<values.shape[1]-1 and 1<=y<values.shape[0]-1):
                continue
            local=values[y-1:y+2,x-1:x+2]
            prob=np.exp(np.clip(local-peak,-40,0))
            offset=np.arange(-1,2,dtype=np.float32)
            px=float(x0+x-4-half+(prob.sum(axis=0)@offset)/prob.sum())
            py=float(y0+y-4-half+(prob.sum(axis=1)@offset)/prob.sum())
            if np.hypot(px,py)<=self.params['max_correction_px']:
                scored.append((peak,px,py))
        if not scored:
            return None
        scored.sort(reverse=True)
        peak,px,py=scored[0]
        if len(scored)>1 and peak-scored[1][0]<self.params['min_gap']:
            return None
        confidence=float(1./(1.+np.exp(-np.clip(peak,-40,40))))
        return dict(center=(float(point[0]+px),float(point[1]+py)),confidence=confidence,
                    method='local_fitted_marker',logit=round(peak,3))
