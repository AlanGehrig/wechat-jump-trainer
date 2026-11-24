"""Locate a resting pawn and the next game platform from a screenshot only."""
import cv2
import numpy as np
from PIL import Image


def _background(rgb):
    # Follow the common colour from row to row. The floor is a smooth vertical gradient.
    rows = rgb[250:760, 4:-4:3].astype(np.int16)
    bg = np.zeros((len(rgb), 3), dtype=np.int16)
    prior = np.median(rows[0], axis=0)
    for i, row in enumerate(rows):
        nearby = np.max(np.abs(row - prior), axis=1) < 5
        if nearby.any():
            prior = np.median(row[nearby], axis=0)
        bg[250+i] = prior
    return bg


def _pawn(rgb):
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    mask = (b-r > 8) & (b-g > 6) & (r < 115) & (g < 110) & (b < 155)
    mask[:365] = False
    mask[800:] = False
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    pieces = stats[1:].tolist()
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
    roi = rgb[roi_top:roi_end].astype(np.int16)
    candidates = []
    for dx, dy in [(0, 5), (0, 11), (-3, 17), (4, 21)]:
        sx, sy = int(np.clip(tx+dx, 3, 539)), ty+dy-roi_top
        if sy >= len(roi) or not foreground[roi_top+sy, sx]:
            continue
        mat = np.max(np.abs(roi-roi[sy,sx]),axis=2) <= 8
        mat &= foreground[roi_top:roi_end]
        mat = cv2.morphologyEx(mat.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mat)
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
    x0,x1=max(4,tx-65),min(539,tx+66)
    y0,y1=max(255,ty-2),min(len(rgb),ty+78)
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
        if not (25<=w<=120 and .40<=h/w<=.71 and abs(ty-y0-y)<=5):
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


def analyze(frame):
    """Return foot and target coordinates in original pixels, confidence and reason."""
    if not isinstance(frame, Image.Image):
        raise TypeError('frame must be a PIL image')
    ow, oh = frame.size
    sx, sy = ow/543, oh/1029
    if abs(sx/sy-1)>.14:
        return dict(foot=None,target=None,confidence=0.,reason='unexpected full-window aspect ratio')
    rgb = np.asarray(frame.convert('RGB').resize((543,1029),Image.Resampling.BILINEAR))
    pawn = _pawn(rgb)
    if not pawn:
        return dict(foot=None,target=None,confidence=0.,reason='resting pawn body and head not found')
    foot, bbox = pawn
    def mapped(pt):
        return (round(pt[0]*sx), round(pt[1]*sy))
    bg = _background(rgb)
    foreground = np.max(np.abs(rgb.astype(np.int16)-bg[:,None,:]),axis=2)>12
    foreground[:270]=False
    foreground[foot[1]-20:]=False
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
        return dict(foot=mapped(foot),target=None,confidence=0.,reason='no upper platform contour')
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
            return dict(foot=mapped(foot),target=mapped((cx,cy)),confidence=round(conf,3),reason=reason)
    return dict(foot=mapped(foot),target=None,confidence=0.,reason='ambiguous or decorated platform outline')
