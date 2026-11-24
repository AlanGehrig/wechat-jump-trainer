import json,time,hashlib
import numpy as np
from train_refinement import load_annotations,evaluate,summaries
from jump_refinement import MarkerRefiner
from review_labels import ROOT,read_img
annotations=load_annotations()
model=MarkerRefiner(); groups={b:[r for r in annotations if r['batch']==b] for b in ['C','D','E','F']}
all_results={k:evaluate(model,v) for k,v in groups.items() if v}
models_sha=hashlib.sha256((ROOT/'marker_weights.json').read_bytes()).hexdigest()
merged=sum(all_results.values(),[])
benchmark=[]
held=[r for r in annotations if r['batch'] in groups]
samples=[(read_img(r),r) for r in held]
for _ in range(5):
 for image,r in samples:
  start=time.perf_counter_ns();model.predict(image,r['coarse'],r['original_box'][:2]);benchmark.append((time.perf_counter_ns()-start)/1e6)
result={'weights_sha256':models_sha,'split':'chronologically later independent scenes','summary':{k:summaries(rows) for k,rows in all_results.items()},
        'combined':summaries(merged),'warm_additional_ms':{'runs':len(benchmark),'p50':round(float(np.median(benchmark)),3),'p95':round(float(np.percentile(benchmark,95)),3)},'rows':merged}
(ROOT/'独立测试.json').write_text(json.dumps(result,ensure_ascii=False,indent=1),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('summary','combined','warm_additional_ms')}))
