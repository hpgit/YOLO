import json, sys
from pathlib import Path
from importlib.metadata import version
import os, subprocess
from collections import Counter
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import YoloDataset, collate_fn
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.model_utils import GradientAccumulation
from lightning.pytorch.loops.optimization.automatic import ClosureResult

torch.set_num_threads(2)
torch.manual_seed(0)
out = {}
row = [0, .5, .5, .2, .2]
out['txt_label'] = {'input': row, 'actual_xyxy': YoloDataset.load_valid_labels(None, 'probe', [row]).tolist(), 'expected_xyxy': [[0,.4,.4,.6,.6]]}
cfg = OmegaConf.load('yolo/config/model/v9-t.yaml')
model = YOLO(cfg)
before = {k: v.clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
Vec2Box(model, cfg.anchor, [64,64], torch.device('cpu'))
after = model.state_dict()
changed = [k for k,v in before.items() if not torch.equal(v,after[k])]
out['auto_stride_bn'] = {'changed_buffers':len(changed), 'examined_buffers':len(before), 'sample': {k: {'before':before[k].flatten()[:3].tolist(), 'after':after[k].flatten()[:3].tolist()} for k in changed[:3]}}

grads=[]
for norm in [1,4]:
    p=torch.tensor(1.,requires_grad=True)
    for _ in range(4): ClosureResult.from_training_step_output(p*16, normalize=norm).closure_loss.backward()
    grads.append(float(p.grad))
out['accumulation_gradient']={'official_sum_example':grads[0], 'lightning_default_example':grads[1]}
cb=GradientAccumulation(OmegaConf.create({'equivalent_batch_size':64,'batch_size':16}),OmegaConf.create({'warmup':{'epochs':3}}))
trainer=SimpleNamespace(world_size=1,global_step=0,accumulate_grad_batches=1)
cb.setup(trainer, SimpleNamespace(train_loader=range(100)), 'fit')
trace=[]
for epoch in range(10):
    cb.on_train_epoch_start(trainer,None)
    start=cb.current_batch
    factors=[]
    for batch in range(100):
        cb.on_train_batch_start(trainer,None)
        factors.append(trainer.accumulate_grad_batches)
        if (batch+1)%trainer.accumulate_grad_batches==0 or batch==99:trainer.global_step+=1
        cb.on_train_batch_end(trainer,None)
    trace.append({'epoch':epoch,'counter_start':start,'factor_start':factors[0],'factor_end':factors[-1]})
out['accumulation_counter_simulation']=trace

data=json.loads(Path('data/coco/annotations/instances_val2017.json').read_text())
counts=Counter(a['image_id'] for a in data['annotations'] if not a['iscrowd'])
diffs=[]
for a in data['annotations']:
    if a['iscrowd'] or not a.get('segmentation') or not isinstance(a['segmentation'],list):continue
    pts=np.array([p for seg in a['segmentation'] for p in np.array(seg).reshape(-1,2)])
    xy=np.r_[pts.min(0),pts.max(0)]
    x,y,w,h=a['bbox']
    diffs.append(float(np.max(np.abs(xy-[x,y,x+w,y+h]))))
out['coco_val_annotations']={'images':len(data['images']),'annotations':len(data['annotations']),'crowd_excluded':sum(bool(a['iscrowd']) for a in data['annotations']), 'noncrowd_images_gt_over100':sum(n>100 for n in counts.values()),'noncrowd_gt_dropped_by_cap':sum(max(0,n-100) for n in counts.values()),'polygon_vs_bbox_gt_compared':len(diffs),'polygon_bbox_diff_gt1px':sum(d>1 for d in diffs),'polygon_bbox_maxdiff_px':max(diffs)}
tiny = YOLO.__new__(YOLO)
torch.nn.Module.__init__(tiny)
tiny.model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
tiny.save_load_weights({'state_dict': {'model.model.0.weight': torch.tensor([[2.]]), 'ema.model.0.weight': torch.tensor([[7.]])}})
out['checkpoint_loader_probe'] = {'regular_weight': 2, 'ema_weight': 7, 'loaded_weight': tiny.model[0].weight.item()}
out['provenance'] = {'local_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'reference_commit': '5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff', 'torch': version('torch'), 'lightning': version('lightning'), 'torchmetrics': version('torchmetrics'), 'scope': 'CPU behavioral probes and COCO val annotation inspection; not AP reproduction'}
Path(__file__).with_name('probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
