"""Verify the revision archive and replay the primary fitted forecasting pipeline."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import warnings

import numpy as np
import pandas as pd


def load(name, path, transform=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source = path.read_text(encoding='utf-8')
    exec(compile(transform(source) if transform else source, str(path), 'exec'), module.__dict__)
    return module


def setup(root):
    folder = root/'r2_experiment_plan_20261018'
    project = root/'CCR_HQFNet_评审修正_物理拥塞路由_20260721'
    sys.path.insert(0, str(folder))
    def relocate(source):
        old = "ROOT=Path('D:/White_Horse_Lake/第二篇论文/第二篇论文基准模型/rolling_v3/real_feature')"
        assert old in source
        source = source.replace(old, 'ROOT=W/"raw_inputs"')
        source = source.replace("Path(meta['adapter'])", "P/Path(meta['adapter'].replace(chr(92),'/').split(P.name+'/')[-1])")
        source = source.replace("meta['source_hashes'][name].lower()", "next(v for k,v in meta['source_hashes'].items() if Path(k.replace(chr(92),'/')).name==Path(name).name).lower()")
        return source
    restore = load('restore_r47_validation', folder/'restore_r47_validation.py', relocate)
    import joint_routing_trial as trial
    return folder, project, restore, trial


def verify(root, output):
    b = root/'r2_experiment_plan_20261018'
    j = b/'joint_routing_trial_20261022'
    frame = pd.read_csv(j/'trial_predictions_LOCAL_ONLY.csv')
    assert not frame.duplicated(['model','time']).any()
    sizes = frame.groupby('model').size()
    assert sizes.eq(6048).all()
    full = frame[frame.model.eq('Full')].sort_values('time')
    assert len(full)==6048 and full.time.nunique()==6048
    assert np.max(abs(full.predicted-full.q50)) < 1e-8
    y, p, lo, hi = (full[k].to_numpy(float) for k in ['actual','predicted','q05','q95'])
    mae = float(abs(y-p).mean())
    wink = hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
    wis = float((.5*abs(y-p)+.05*wink).mean()/1.5)
    assert abs(mae-59.55745866961517)<1e-8
    assert abs(wis-(.5*mae+.05*wink.mean())/1.5)<1e-10
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model,g in frame.groupby('model'):
        e=g.actual-g.predicted
        rows.append(dict(model=model,n=len(g),MAE=float(abs(e).mean()),RMSE=float(np.sqrt((e*e).mean()))))
    pd.DataFrame(rows).to_csv(output/'primary_components.csv',index=False)
    sys.path.insert(0,str(b))
    probability=load('revision_probability',b/'complete_matched_capacity_and_probability_20261023.py')
    probability.OUT=output
    probability.probabilities()
    probability.risk_uncertainty()
    expected=pd.read_csv(b/'four_comment_experiment_closeout_20261023/consistent_probability_tables.csv')
    actual=pd.read_csv(output/'consistent_probability_tables.csv')
    pd.testing.assert_frame_equal(expected,actual,check_exact=False,rtol=1e-10,atol=1e-9)
    summary=dict(primary_targets=6048,primary_days=full.delivery_day.nunique(),primary_models=len(sizes),
        Full_MAE=mae,WIS=wis,MPIW90=float((hi-lo).mean()),
        probability_recalculation='PASS',quantile_point_identity='PASS')
    (output/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


def replay(root,output):
    folder,project,restore,trial=setup(root)
    before=pd.read_csv(trial.OUT/'trial_predictions_LOCAL_ONLY.csv').sort_values(['model','time']).reset_index(drop=True)
    evaluator=load('revision_evaluator',folder/'evaluate_joint_routing_trial.py')
    evaluator.evaluate()
    after=pd.read_csv(trial.OUT/'trial_predictions_LOCAL_ONLY.csv').sort_values(['model','time']).reset_index(drop=True)
    pd.testing.assert_frame_equal(before,after,check_exact=False,rtol=1e-9,atol=1e-7)
    print('PASS: fitted-model replay matches all archived primary pathways.')
    verify(root,output)


def train(root,output,limit):
    folder,project,restore,trial=setup(root)
    reference=json.loads((trial.OUT/'protocol.json').read_text(encoding='utf-8'))
    trial.OUT=output.resolve()
    def protocol():
        trial.OUT.mkdir(parents=True,exist_ok=True)
        reference['source_sha256']=hashlib.sha256(Path(trial.__file__).read_bytes()).hexdigest()
        (trial.OUT/'protocol.json').write_text(json.dumps(reference,indent=2),encoding='utf-8')
        return reference
    trial.make_protocol=protocol
    trial.select(limit)
    print('Completed downstream fitting in',trial.OUT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/revision_reproduction'))
    p.add_argument('--mode',choices=['verify','replay','train-downstream'],default='verify')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    if a.mode=='verify':verify(a.archive.resolve(),a.output)
    elif a.mode=='replay':replay(a.archive.resolve(),a.output)
    else:train(a.archive.resolve(),a.output,a.limit)

if __name__=='__main__':main()
