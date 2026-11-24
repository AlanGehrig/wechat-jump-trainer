"""Read bonus platforms from unobstructed pre-jump screenshots of WeChat Jump Jump.

The caller should carry a recognized target across ONE verified successful jump.
Only after landing should it apply the corresponding WAIT_SECONDS and read the HUD.
"""
import cv2
import numpy as np
from PIL import Image

WAIT_SECONDS = {"record_player": 3.0, "rubiks_cube": 3.0, "wechat_pay": 3.0}
BONUS_POINTS = {"record_player": 30, "rubiks_cube": 10, "wechat_pay": None}


def _regions(mask, low, high):
    binary=mask.view(np.uint8)
    if cv2.countNonZero(binary)<low:
        return []
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary)
    return [(tuple(map(int, stats[i, :4])), tuple(centers[i]))
            for i in range(1, count) if low <= stats[i, 4] <= high]


def _clamp_slice(mask, x, y, w, h):
    return mask[max(0,round(y)):min(mask.shape[0],round(y+h)),
                max(0,round(x)):min(mask.shape[1],round(x+w))]


def _centers(rgb):
    red,green,blue=cv2.split(rgb.astype(np.int16))
    gray=(np.maximum(np.maximum(red,green),blue)-np.minimum(np.minimum(red,green),blue))<=4
    # The red and gold stamp is concentric with a gray vinyl record.
    charcoal = gray & (red >= 116) & (red <= 120)
    for (x,y,w,h), _ in _regions(charcoal, 850, 11500):
        if not (52 <= w <= 190 and 0.47 <= h / w <= 0.69):
            continue
        cx, cy = x+(w-1)/2, y+(h-1)/2
        patch = _clamp_slice(rgb,cx-0.30*w,cy-0.30*h,0.60*w,0.60*h)
        r,g,b=cv2.split(patch.astype(np.int16))
        center_stamp = (r>g+27)&(g>b+5)&(r>155)&(g<225)&(b>55)
        if center_stamp.size and center_stamp.mean() >= 0.24:
            # The game draws its center marker just above the printed label.
            yield "record_player", cx, cy-0.08*h

    # WeChat Pay has a matte silver diamond and a separate green projecting sign.
    silver = gray & (red >= 220) & (red <= 226)
    for (x,y,w,h), _ in _regions(silver, 3900, 30000):
        if not (98 <= w <= 296 and 0.47 <= h/w <= 0.71):
            continue
        cx,cy = x+(w-1)/2, y+(h-1)/2
        patch=_clamp_slice(rgb,cx-0.425*w,cy+0.14*h,0.245*w,0.47*h)
        if not patch.size:
            continue
        r,g,b=cv2.split(patch.astype(np.int16))
        left_sign=(g>r+38)&(g>b+27)&(g>120)&(r>38)
        if left_sign.size and left_sign.mean() > 0.28:
            yield "wechat_pay", cx, cy-0.028*h

    # The cube top has nine disjoint blue diamonds; a single diamond's center
    # can be mistaken for the center of the entire platform in the general model.
    blue_tiles = (blue>red+85) & (blue>green+48) & (green>red+15)
    tiles = [(cx,cy,w,h) for (x,y,w,h),(cx,cy) in _regions(blue_tiles, 90, 2000)
             if 15 <= w <= 78 and 0.43 <= h/w <= 0.73]
    for cx,cy,w,h in tiles:
        dx,dy = w*0.60,h*0.59
        offsets = ((0,0),(-1,-1),(1,-1),(-1,1),(1,1),(0,-2),(0,2),(-2,0),(2,0))
        def has_tile(i,j):
            return any(abs(tx-cx-i*dx)<w*0.20 and abs(ty-cy-j*dy)<h*0.23
                       and 0.78<tw/w<1.23 and 0.77<th/h<1.25 for tx,ty,tw,th in tiles)
        if all(has_tile(i,j) for i,j in offsets):
            yield "rubiks_cube", cx, cy


def locate_bonus_target(frame, scene):
    """Return (kind, (x, y)) in original frame coordinates, or None."""
    if not isinstance(frame, Image.Image) or not scene or scene.get("foot") is None:
        return None
    sx, sy = frame.width/543, frame.height/1029
    if abs(sx/sy-1)>0.055:
        return None
    if frame.mode!="RGB":
        frame=frame.convert("RGB")
    if frame.size!=(543,1029):
        frame=frame.resize((543,1029),Image.Resampling.BILINEAR)
    fx, fy = scene["foot"][0]/sx, scene["foot"][1]/sy
    # Slightly below the pawn is retained to capture the corner of a near platform.
    y0, y1 = 260, min(900,round(fy+85))
    if y1<=y0:
        return None
    target = scene.get("target")
    matches = []
    x0,x1=0,543
    if target:
        tx,ty=target[0]/sx,target[1]/sy
        # The largest accepted silver face is 296px wide; 180px also includes
        # its side sign and the full extent of every legal cube/record candidate.
        x0,x1=max(0,int(tx)-180),min(543,int(tx)+181)
        y0,y1=max(y0,int(ty)-150),min(y1,int(ty)+180)
        if x1<=x0 or y1<=y0:
            return None
    rgb=np.asarray(frame.crop((x0,y0,x1,y1)))
    for kind,x,y in _centers(rgb):
        x += x0
        y += y0
        if y>=fy-24 or abs(x-fx)<32:
            continue
        if target:
            tx,ty = target[0]/sx,target[1]/sy
            if abs(x-tx)>20 or abs(y-ty)>(48 if kind=="rubiks_cube" else 18):
                continue
            # The actual white marker takes priority over the graphic in this case.
            if scene.get("reason")=="center disk on upper platform":
                x,y=tx,ty
        matches.append((kind,(round(x*sx),round(y*sy))))
    return matches[0] if len(matches)==1 else None


def classify_target(frame, scene):
    result = locate_bonus_target(frame, scene)
    return result[0] if result else None
