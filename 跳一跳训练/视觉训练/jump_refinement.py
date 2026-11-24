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
    for bandwidth in (2.,8.,22.):
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
        self.weights=np.array(self.params['weights'],dtype=np.float32)

    def predict(self,image,point,origin=(0,0)):
        patch=extract_patch(image,point,origin)
        if patch is None:
            return None
        features=feature_map(patch,self.params['prototype'])
        scores=features@self.weights+self.params['intercept']
        y,x=np.unravel_index(np.argmax(scores),scores.shape)
        peak=float(scores[y,x])
        minimum=self.params['min_logit']
        if peak<minimum or not (1<=x<scores.shape[1]-1 and 1<=y<scores.shape[0]-1):
            return None
        # Reject isolated edges and competing centers. Return subpixel location
        # from a smooth maximum of the learned local-score surface.
        lo_y,hi_y=max(0,y-5),min(scores.shape[0],y+6)
        lo_x,hi_x=max(0,x-7),min(scores.shape[1],x+8)
        outside=scores.copy()
        outside[lo_y:hi_y,lo_x:hi_x]=-np.inf
        if peak-float(outside.max())<self.params['min_gap']:
            return None
        small=scores[y-1:y+2,x-1:x+2]
        prob=np.exp(np.clip(small-peak,-40,0))
        offset=np.arange(-1,2,dtype=np.float32)
        px=float(x+(prob.sum(axis=0)@offset)/prob.sum()-RADIUS)
        py=float(y+(prob.sum(axis=1)@offset)/prob.sum()-RADIUS)
        if np.hypot(px,py) > self.params['max_correction_px']:
            return None
        confidence=float(1./(1.+np.exp(-np.clip(peak,-40,40))))
        return dict(center=(float(point[0]+px),float(point[1]+py)),confidence=confidence,
                    method='local_fitted_marker',logit=round(peak,3))
