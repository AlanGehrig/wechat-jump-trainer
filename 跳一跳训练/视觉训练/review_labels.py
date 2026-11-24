"""Measure reviewed visible marks with an independent connected-component centroid."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
def load_rows():
    return [json.loads(i) for i in (ROOT/'样本.jsonl').read_text(encoding='utf-8').splitlines()]
def read_img(row):
    return Image.open(ROOT/'画面'/(row['id']+'.png')).convert('RGB')
def make_annotations(positive_ids, reviewed):
    labels=[]
    seen=set()
    for row in load_rows():
        k=int(row['id'][1:])
        if k>reviewed:
            continue
        key=(row['score'],*row['foot'])
        if key in seen:
            labels.append(dict(id=row['id'],status='repeat_scene'))
            continue
        seen.add(key)
        if k not in positive_ids:
            labels.append(dict(id=row['id'],status='reviewed_no_marker'))
            continue
        img=np.asarray(read_img(row)).astype(np.int16)
        cx,cy=np.array(row['coarse'])-row['original_box'][:2]
        x0,y0=cx-42,cy-37
        patch=img[y0:cy+38,x0:cx+43]
        options=[]
        for tolerance in (3,1,0):
            mask=(np.max(np.abs(patch-245),axis=2)<=tolerance).astype(np.uint8)
            _,lab,st,ct=cv2.connectedComponentsWithStats(mask)
            options=[]
            for idx,(x,y,w,h,area) in enumerate(st[1:],1):
                if 13<=w<=22 and 7<=h<=14 and .43<=h/w<=.76 and .57<=area/(h*w)<=.96:
                    u,v=ct[idx]
                    options.append([round(u+x0+row['original_box'][0],3),round(v+y0+row['original_box'][1],3)])
            if len(options)==1:
                break
        if len(options)==1:
            labels.append(dict(id=row['id'],status='reviewed_marker',ref=options[0],
                               reference='Visually identified disk; centroid of its measured colour boundary'))
        else:
            labels.append(dict(id=row['id'],status='uncertain_marker',options=options))
    (ROOT/'标注.json').write_text(json.dumps(labels,indent=1,ensure_ascii=False),encoding='utf-8')
    from collections import Counter
    print(Counter(r['status'] for r in labels))
if __name__=='__main__':
    review=json.loads((ROOT/'人工阅图.json').read_text(encoding='utf-8-sig'))
    make_annotations(set(review['positive_ids']),review['through'])
