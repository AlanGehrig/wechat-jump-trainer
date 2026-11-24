"""Locate a resting pawn and the next game platform from a screenshot only."""
import hashlib
import math
from pathlib import Path
import sys
import time
from types import ModuleType

import cv2
import numpy as np
from PIL import Image


def _background(rgb, stop=760):
    # Most floor pixels share an exact colour on a row. Estimate candidate
    # medians in bulk, then check their defining recurrence in scan order.
    rows = np.ascontiguousarray(rgb[250:stop, 4:-4:3].transpose(0,2,1)).astype(np.int16)
    bg = np.zeros((len(rgb),3),dtype=np.int16)
    first = np.median(rows[0],axis=1)
    coarse = np.median(rows[:,:,::8],axis=2)
    guesses = np.vstack((first,coarse[:-1]))
    prior = first
    for iteration in range(2):
        limits = guesses[:,:,None]
        near = ((rows>limits-5) & (rows<limits+5)).all(axis=1)
        counts = near.sum(axis=1)
        sorted_pixels = np.sort(np.where(near[:,None,:], rows, 256),axis=2)
        index = np.arange(len(rows))[:,None]
        channels = np.arange(3)[None,:]
        left = np.maximum(0,counts-1)//2
        right = counts//2
        medians = (sorted_pixels[index,channels,left[:,None]]+
                   sorted_pixels[index,channels,right[:,None]])*.5
        medians[counts==0] = guesses[counts==0]
        if iteration == 0:
            guesses = np.vstack((first,medians[:-1]))
    # An unverified proposal can arise when a wide platform covers most of
    # a row. Recalculate that one from its exact predecessor.
    matches = np.all(guesses[1:]==medians[:-1],axis=1)
    bg[250:stop] = medians
    k=1
    while k<len(rows):
        bad = np.flatnonzero(~matches[k-1:])
        if not len(bad):
            break
        k += int(bad[0])
        prior=medians[k-1]
        while k<len(rows) and not np.array_equal(prior,guesses[k]):
            r=rows[k]
            valid=np.all((r>prior[:,None]-5)&(r<prior[:,None]+5),axis=0)
            if valid.any():
                prior=np.median(r[:,valid],axis=1)
            medians[k]=prior
            bg[250+k]=prior
            k+=1
        if k<len(rows):
            k+=1
    return bg


def _pawn(rgb):
    r,g,b = cv2.split(rgb[365:800].astype(np.int16))
    mask = (b-r > 8) & (b-g > 6) & (r < 115) & (g < 110) & (b < 155)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    pieces = stats[1:].copy()
    pieces[:,1] += 365
    pieces = pieces.tolist()
    for x, y, w, h, area in sorted(pieces, key=lambda p: -p[-1]):
        if not (27 <= w <= 47 and 45 <= h <= 78 and 800 <= area <= 3300):
            continue
        cx = x + (w-1)*0.5
        for hx, hy, hw, hh, ha in pieces:
            if (20 <= hw <= 35 and 20 <= hh <= 35 and 200 < ha < 1200
                    and abs(hx+hw*0.5-cx)<4 and -2 <= y-hy-hh <= 16):
                return (int(round(cx)), int(round(y+h-1 - w*0.14))), (x-7, hy-3, x+w+7, y+h+2)
    return None


def _top_face(rgb, foreground, tip, foot):
    tx, ty = tip
    # Pick a connected component of the top material, using multiple seeds in
    # case of printed designs. Its aspect ratio checks diamond and round tops.
    roi_top = max(255, ty-2)
    roi_end = min(760, ty+150, foot[1]+12)
    roi = rgb[roi_top:roi_end]
    seen = {}
    candidates = []
    for dx, dy in [(0, 5), (0, 11), (-3, 17), (4, 21)]:
        sx, sy = int(np.clip(tx+dx, 3, 539)), ty+dy-roi_top
        if sy >= len(roi) or not foreground[roi_top+sy, sx]:
            continue
        color=tuple(map(int,roi[sy,sx]))
        if color not in seen:
            lower=tuple(max(0,c-8) for c in color)
            upper=tuple(min(255,c+8) for c in color)
            mat = cv2.inRange(roi,lower,upper)
            mat &= foreground[roi_top:roi_end].view(np.uint8)
            mat = cv2.morphologyEx(mat,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            _,labels,stats,_ = cv2.connectedComponentsWithStats(mat)
            seen[color]=(labels,stats)
        labels,stats=seen[color]
        lab = labels[sy,sx]
        if not lab:
            continue
        x, y, w, h, area = [int(k) for k in stats[lab]]
        if not (45 <= w <= 295 and 19 <= h <= 145):
            continue
        cx, cy = x+(w-1)*.5, roi_top+y+(h-1)*.5
        fill = area/(w*h)
        if abs(cx-tx) > max(14, .20*w) or roi_top+y-ty > 8:
            continue
        aspect_error = abs(h/w - .577)
        if aspect_error > .18 or not (.35 < fill < .91):
            continue
        conf = .98 - .65*aspect_error - .25*abs(cx-tx)/w
        candidates.append((conf,cx,cy,'top material outline'))
    return max(candidates,default=None)


def _center_mark(rgb, fg, tip, foot):
    # Several special platforms draw a small pale disk on their top centre.
    tx, ty = tip
    y0, y1 = ty+8, min(ty+90, foot[1]-22)
    x0, x1 = max(4, tx-30), min(539, tx+31)
    if y1-y0 < 8 or x1-x0 < 14:
        return None
    patch = rgb[y0:y1,x0:x1].astype(np.int16)
    disk = (patch.min(axis=2) >= 232) & (np.ptp(patch,axis=2) <= 7)
    disk &= fg[y0:y1,x0:x1]
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(disk.astype(np.uint8))
    candidates = []
    for i in range(1,len(stats)):
        x, y, w, h, area = [int(n) for n in stats[i]]
        if not (13<=w<=25 and 7<=h<=15 and .43<=h/w<=.72 and .63<=area/(w*h)<=.93):
            continue
        cx,cy = np.rint(centroids[i] + (x0,y0)).astype(int)
        if abs(cx-tx)>15 or abs(abs(cx-foot[0])*.577-(foot[1]-cy))>45:
            continue
        # The disk must be brighter than the continuous top material around it;
        # passing here rejects pale floor gaps and cut-off lettering on a side.
        nbrs=((cx-w//2-3,cy),(cx+w//2+3,cy),(cx,cy-h//2-3),(cx,cy+h//2+3))
        if not all(4<=xx<=538 and fg[yy,xx] for xx,yy in nbrs):
            continue
        ring=np.array([rgb[yy,xx] for xx,yy in nbrs]).astype(np.int16)
        center=np.median(patch[labels==i],axis=0)
        # Printed designs surround the disk with different inks. Their RGB
        # colours may differ as long as every sample clearly separates from it.
        if np.max(np.ptp(ring,axis=0))>20:
            if not (16<=w<=22 and 8<=h<=13 and np.all(np.max(center-ring,axis=1)>=16)):
                continue
        if np.mean(center-ring.mean(axis=0))<12:
            continue
        candidates.append((.97, int(cx), int(cy), 'center disk on upper platform'))
    if len(candidates)==1:
        return candidates[0]
    return None


def _ring_top(rgb, fg, tip):
    tx, ty = tip
    # A silver pillar has concentric bright, dark, then bright ellipses.
    # The nested contours preserve its geometric centre even on a tiny top.
    x0,x1=max(4,tx-82),min(539,tx+83)
    y0,y1=max(255,ty-2),min(len(rgb),ty+94)
    roi=rgb[y0:y1,x0:x1].astype(np.int16)
    bright=(roi.min(axis=2)>=231) & (np.ptp(roi,axis=2)<=8) & fg[y0:y1,x0:x1]
    contours,links=cv2.findContours(bright.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if links is None:
        return None
    for i,outer in enumerate(contours):
        hole=links[0,i,2]
        if hole<0:
            continue
        core=links[0,hole,2]
        if core<0:
            continue
        x,y,w,h=cv2.boundingRect(outer)
        if not (25<=w<=150 and .40<=h/w<=.71 and abs(ty-y0-y)<=5):
            continue
        cx,cy=x+(w-1)/2,y+(h-1)/2
        ix,iy,iw,ih=cv2.boundingRect(contours[core])
        if not (.32<=iw/w<=.85 and .32<=ih/h<=.85):
            continue
        if abs(ix+(iw-1)/2-cx)>3 or abs(iy+(ih-1)/2-cy)>3 or abs(x0+cx-tx)>5:
            continue
        return (.975,x0+cx,y0+cy,'concentric cylinder rim')
    return None


def _record_label(rgb, fg, tip):
    tx, ty = tip
    # The printed red label stays concentric with the dark vinyl when the
    # separate white landing mark disappears. Four neutral rim samples confirm it.
    x0, x1 = max(4,tx-60), min(539,tx+61)
    y0, y1 = ty+8, min(ty+108,len(rgb))
    patch = rgb[y0:y1,x0:x1].astype(np.int16)
    r,g,b = patch.transpose(2,0,1)
    ink = (r>180) & (g<207) & (r-g>32) & (r-b>32) & fg[y0:y1,x0:x1]
    contours,_ = cv2.findContours(ink.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if not (46<=w<=85 and .50<=h/w<=.65 and .69<=cv2.contourArea(contour)/(w*h)<=.83):
            continue
        cx,cy = round(x0+x+(w-1)*.5), round(y0+y+(h-1)*.5)
        if abs(cx-tx)>8 or not 24<=cy-ty<=85:
            continue
        dx,dy=round(.65*w),round(.67*h)
        if cx-dx<4 or cx+dx>538:
            continue
        rim=np.array([rgb[cy,cx-dx],rgb[cy,cx+dx],rgb[cy-dy,cx],rgb[cy+dy,cx]]).astype(np.int16)
        if rim.max()<=158 and rim.min()>=50 and np.ptp(rim,axis=1).max()<=12 and np.ptp(rim,axis=0).max()<=20:
            return (.97,cx,cy,'printed label inside vinyl rim')
    return None


def _light_platter(rgb, fg, tip):
    # The grey device carries an off-white platter around a smaller grey hub.
    # Their two separate fitted ellipses agree on the landing point.
    tx, ty = tip
    x0, x1 = max(4,tx-95),min(539,tx+96)
    y0, y1 = ty+2,min(len(rgb),ty+117)
    roi = rgb[y0:y1,x0:x1].astype(np.int16)
    light=(roi.min(axis=2)>202) & (np.ptp(roi,axis=2)<=7)
    contours,links=cv2.findContours(light.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if links is None:
        return None
    for i,outer in enumerate(contours):
        hub=links[0,i,2]
        if hub<0:
            continue
        x,y,w,h=cv2.boundingRect(outer)
        if not (56<=w<=145 and .47<=h/w<=.67 and .69<=cv2.contourArea(outer)/(w*h)<=.84):
            continue
        cx,cy=x+(w-1)/2,y+(h-1)/2
        ix,iy,iw,ih=cv2.boundingRect(contours[hub])
        if not (.23<=iw/w<=.40 and .22<=ih/h<=.42 and abs(cx-(ix+(iw-1)/2))<=3 and abs(cy-(iy+(ih-1)/2))<=3):
            continue
        px,py=x0+round(cx),y0+round(cy)
        if abs(px-tx)>23 or not 18<=py-ty<=80:
            continue
        dx=round(w*.32)
        rim=rgb[py,[px-dx,px+dx]].astype(np.int16)
        center=rgb[py,px].astype(np.int16)
        if not (fg[py,px] and np.ptp(center)<=8 and 22<=np.mean(rim-center)<=100 and np.ptp(rim,axis=1).max()<=8):
            continue
        return (.975,x0+cx,y0+cy,'pale platter around a grey hub')
    return None


def _silhouette(fg, tip, foot):
    tx,ty = tip
    # Follow the right silhouette: lighting casts the ground shadow to the left.
    edge = []
    for y in range(ty+1,min(ty+110, foot[1])):
        row = fg[y]
        start = int(np.clip(tx, 5, 537))
        if not row[start]:
            break
        gap = np.flatnonzero(~row[start:540])
        right = start + int(gap[0]-1) if len(gap) else 539
        edge.append((y,right))
    for i in range(9,len(edge)-8):
        y,right=edge[i]
        if right-tx < 18 or right >= 537:
            continue
        tail = np.array([p[1] for p in edge[i:i+8]])
        if tail.max()-tail.min() > 2 or abs(tail.mean()-right)>1:
            continue
        # Most platforms have diamond or ellipse tops, both with the same
        # projection. A flat side begins close to the top face's middle row.
        predicted_height=(right-tx)*.577
        if abs(y-ty-predicted_height) > max(12,predicted_height*.35):
            continue
        return (.82,tx,(y+ty+predicted_height)/2,'outer contour and right rim')
    return None


# Optional local model: the public target stays geometric, while matched marks
# are separate observations for a controller's explicit horizontal-only trial.
_MARKER_ROOT = Path(__file__).resolve().parent / "跳一跳训练" / "视觉训练"
_MARKER_CACHE = {"next_check": 0., "stamp": None, "model": None, "version": None}
_MARKER_METHOD = "local_fitted_marker"


def _marker_model():
    cache = _MARKER_CACHE
    when = time.monotonic()
    if when >= cache["next_check"]:
        cache["next_check"] = when + 2.
        try:
            code = _MARKER_ROOT / "jump_refinement.py"
            params = _MARKER_ROOT / "marker_weights.json"
            files = (code, params)
            stamp = tuple((st.st_mtime_ns, st.st_size) for st in (f.stat() for f in files))
            if stamp != cache["stamp"]:
                source, weights = (f.read_bytes() for f in files)
                mod = ModuleType("_own_jump_marker")
                mod.__file__ = str(code)
                exec(compile(source, str(code), "exec"), mod.__dict__)
                model = mod.MarkerRefiner(params)
                if (not np.isfinite(model.weights).all() or not model.weights.size or
                        weights != params.read_bytes()):
                    raise ValueError("Incomplete marker update")
                version = ("marker-" + hashlib.sha256(weights).hexdigest()[:16] + "-" +
                           hashlib.sha256(source).hexdigest()[:12])
                cache.update(stamp=stamp, model=model, version=version)
        except (OSError, ValueError, KeyError, ImportError, AttributeError, SyntaxError):
            # A previous complete version can serve during a two-file update.
            pass
    return cache["model"], cache["version"]


def _scene(foot, target, confidence, reason):
    return dict(foot=foot, target=target, confidence=confidence, reason=reason,
                geometry_target=target, marker_target=None, marker_aim_target=None,
                marker_confidence=None, marker_logit=None, marker_method=None,
                marker_version=None, marker_updated=False, marker_status="ineligible_geometry")


def _annotate_marker(scene, rgb, canonical_frame, point, sx, sy):
    if scene["reason"] != "top material outline" or scene["confidence"] < .93:
        return scene
    model, version = _marker_model()
    scene.update(marker_method=_MARKER_METHOD, marker_version=version)
    if model is None:
        scene["marker_status"] = "model_unavailable"
        return scene
    try:
        # The learned filters are applied once to this ROI in canonical coordinates.
        match = model.predict(rgb, point)
    except (ValueError, TypeError, ArithmeticError, IndexError, cv2.error):
        scene["marker_status"] = "model_abstained"
        return scene
    if not match:
        scene["marker_status"] = "no_reliable_marker"
        return scene
    mx, my = match["center"]
    dx, dy = mx-point[0], my-point[1]
    confidence, logit = match["confidence"], match["logit"]
    if not all(math.isfinite(v) for v in (mx,my,confidence,logit)):
        scene["marker_status"] = "model_abstained"
        return scene
    scene.update(marker_confidence=round(confidence,7), marker_logit=logit)
    if (not (4<=mx<=538 and 270<=my<760) or abs(dx)>12 or abs(dy)>9 or
            math.hypot(dx,dy)>15):
        scene["marker_status"] = "geometry_disagreement"
        return scene
    # The game runner puts its live bonus reader in sys.modules. Use its
    # classification also when a decorated block superficially looks ordinary.
    try:
        bonus = sys.modules.get("jump_bonus")
        if bonus is None:
            import jump_bonus as bonus
        canonical_scene = dict(scene, foot=(scene["foot"][0]/sx,scene["foot"][1]/sy),
                               target=point)
        if bonus.locate_bonus_target(canonical_frame,canonical_scene) is not None:
            scene["marker_status"] = "special_platform"
            return scene
    except (ImportError, AttributeError, ValueError, TypeError, cv2.error):
        scene["marker_status"] = "platform_type_unverified"
        return scene
    mx, my = round(mx*sx,3), round(my*sy,3)
    scene.update(marker_target=(mx,my), marker_aim_target=(mx,scene["geometry_target"][1]),
                 marker_status="accepted", marker_updated=True)
    return scene


def analyze(frame):
    """Return foot and target coordinates in original pixels, confidence and reason."""
    if not isinstance(frame, Image.Image):
        raise TypeError('frame must be a PIL image')
    ow, oh = frame.size
    sx, sy = ow/543, oh/1029
    if abs(sx/sy-1)>.14:
        return _scene(foot=None,target=None,confidence=0.,reason='unexpected full-window aspect ratio')
    if frame.mode!='RGB':
        frame=frame.convert('RGB')
    if frame.size!=(543,1029):
        frame=frame.resize((543,1029),Image.Resampling.BILINEAR)
    rgb=np.asarray(frame)
    pawn = _pawn(rgb)
    if not pawn:
        return _scene(foot=None,target=None,confidence=0.,reason='resting pawn body and head not found')
    foot, bbox = pawn
    def mapped(pt):
        return (round(pt[0]*sx), round(pt[1]*sy))
    stop=foot[1]-20
    bg = _background(rgb, min(stop,760))
    foreground = np.zeros(rgb.shape[:2], dtype=bool)
    band=rgb[270:stop]
    row_colors=np.broadcast_to(bg[270:stop,None,:].astype(np.uint8),band.shape)
    diff=cv2.absdiff(band,np.ascontiguousarray(row_colors))
    foreground[270:stop]=cv2.inRange(diff,(0,0,0),(12,12,12))==0
    foreground[:,:4]=False
    foreground[:,539:]=False
    x0,y0,x1,y1=bbox
    foreground[y0:y1,x0:x1]=False
    # Ignore HUDs and small particles. The first durable row section is the
    # apex of the highest new platform when the camera has settled.
    opened=cv2.morphologyEx(foreground.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n, labels, stats, _=cv2.connectedComponentsWithStats(opened)
    tips=[]
    for lab in range(1,n):
        x,y,w,h,area=[int(v) for v in stats[lab]]
        if area < 400 or w < 32 or h < 18:
            continue
        ys, xs=np.nonzero(labels[y:y+3]==lab)
        if not len(xs):
            continue
        tipx=int(round(np.median(xs)))
        if abs(tipx-foot[0])<35:
            continue
        tips.append((y,tipx))
    if not tips:
        return _scene(foot=mapped(foot),target=None,confidence=0.,reason='no upper platform contour')
    for y,tx in sorted(tips):
        if y <= 272:
            continue
        tip=(tx,y)
        candidate=(_top_face(rgb,foreground,tip,foot) or _center_mark(rgb,foreground,tip,foot)
                   or _ring_top(rgb,foreground,tip) or _record_label(rgb,foreground,tip)
                   or _light_platter(rgb,foreground,tip) or _silhouette(foreground,tip,foot))
        if candidate:
            conf, cx, cy, reason=candidate
            if cy >= foot[1]-18:
                continue
            candidate=(int(round(cx)),int(round(cy)))
            scene=_scene(foot=mapped(foot),target=mapped((cx,cy)),confidence=round(conf,3),reason=reason)
            return _annotate_marker(scene,rgb,frame,candidate,sx,sy)
    return _scene(foot=mapped(foot),target=None,confidence=0.,reason='ambiguous or decorated platform outline')
