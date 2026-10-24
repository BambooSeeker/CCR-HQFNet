"""Verify the revision archive and replay the primary fitted forecasting pipeline."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import warnings

# Imported archive sources must remain byte-for-byte unchanged.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


def replace_once(source, before, after, changes, reason):
    count = source.count(before)
    if count != 1:
        raise ValueError(f'Expected one source match for {reason}, found {count}')
    changes.append(dict(reason=reason, before=before, after=after))
    return source.replace(before, after, 1)


def load(name, path, transform=None, audit=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source = path.read_text(encoding='utf-8')
    changes = []
    transformed = transform(source, changes) if transform else source
    if audit is not None:
        audit.append(dict(module=name, path=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            executed_source_sha256=hashlib.sha256(transformed.encode('utf-8')).hexdigest(),
            changes=changes))
    exec(compile(transformed, str(path), 'exec'), module.__dict__)
    return module


def prepare_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('--output must be separate from the frozen archive')
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_config')
    return root, output


def setup(root, audit=None):
    folder = root/'r2_experiment_plan_20261018'
    project = root/'CCR_HQFNet_评审修正_物理拥塞路由_20260721'
    sys.path.insert(0, str(folder))
    def relocate(source, changes):
        old = "ROOT=Path('D:/White_Horse_Lake/第二篇论文/第二篇论文基准模型/rolling_v3/real_feature')"
        source = replace_once(source, old, 'ROOT=W/"raw_inputs"', changes,
                              'read historical inputs from the release')
        source = replace_once(source, "Path(meta['adapter'])", "P/Path(meta['adapter'].replace(chr(92),'/').split(P.name+'/')[-1])",
                              changes, 'relocate adapter paths without changing their identity')
        source = replace_once(source, "meta['source_hashes'][name].lower()", "next(v for k,v in meta['source_hashes'].items() if Path(k.replace(chr(92),'/')).name==Path(name).name).lower()",
                              changes, 'compare archived input hashes after path relocation')
        return source
    restore = load('restore_r47_validation', folder/'restore_r47_validation.py', relocate, audit)
    load('evaluate_congestion_shrinkage', folder/'evaluate_congestion_shrinkage.py', audit=audit)
    trial = load('joint_routing_trial', folder/'joint_routing_trial.py', audit=audit)
    return folder, project, restore, trial


def verify(root, output):
    root, output = prepare_output(root, output)
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
    root, output = prepare_output(root, output)
    audit = []
    status = dict(status='running', archive=str(root), output=str(output), sources=audit,
                  model_training=False, author_local_documents_read=False)
    try:
        folder,project,restore,trial=setup(root, audit)
        archive_trial = trial.OUT
        before=pd.read_csv(archive_trial/'trial_predictions_LOCAL_ONLY.csv').sort_values(['model','time']).reset_index(drop=True)
        def relocate_evaluator(source, changes):
            edits = [
                ('O=trial.OUT', 'ARCHIVE_TRIAL=trial.OUT\nO=REPLAY_OUTPUT',
                 'separate immutable archive reads from regenerated output writes'),
                ("frozen_path=O/'all_selection_frozen.json'", "frozen_path=ARCHIVE_TRIAL/'all_selection_frozen.json'",
                 'read the frozen selection from the archive'),
                ("protocol=json.loads((O/'protocol.json').read_text())", "protocol=json.loads((ARCHIVE_TRIAL/'protocol.json').read_text())",
                 'read the archived protocol'),
                ("sha(O/'protocol.json')", "sha(ARCHIVE_TRIAL/'protocol.json')",
                 'retain the archived protocol hash check'),
                ("dest=O/'folds'/key", "dest=ARCHIVE_TRIAL/'folds'/key",
                 'load saved estimators and calibration records without copying or refitting'),
                ("(O/'evaluation_before_carrier_floor_fix_LOCAL_ONLY')", "(ARCHIVE_TRIAL/'evaluation_before_carrier_floor_fix_LOCAL_ONLY')",
                 'retain the archive provenance lookup'),
                ("    hashes={str(path):sha(path) for path in [\n"
                 "        Path('D:/White_Horse_Lake/第二篇论文/revise稿件文件/重新起点_第二轮修改前/Revised manuscript.docx'),\n"
                 "        Path('D:/White_Horse_Lake/第二篇论文/revise稿件文件/Revised manuscript (unmarked, editable source file).docx')]}\n"
                 "    assert all(h==protocol['protected_manuscript_sha256'] for h in hashes.values())",
                 "    hashes={}\n"
                 "    # Historical author-document protection is not a forecasting input.\n"
                 "    # Its removal is recorded in portability_audit.json; archive checks remain.",
                 'omit the historical author-local DOCX guard, not a scientific integrity check'),
            ]
            for before_text, after_text, reason in edits:
                source = replace_once(source, before_text, after_text, changes, reason)
            return source
        # Inject only the output destination; archived source and model files stay untouched.
        def transform(source, changes):
            source = relocate_evaluator(source, changes)
            return replace_once(source, 'O=REPLAY_OUTPUT', f'O=Path({str(output)!r})',
                                changes, 'use the caller-specified output directory')
        evaluator=load('revision_evaluator',folder/'evaluate_joint_routing_trial.py', transform, audit)
        evaluator.evaluate()
        after=pd.read_csv(output/'trial_predictions_LOCAL_ONLY.csv').sort_values(['model','time']).reset_index(drop=True)
        pd.testing.assert_frame_equal(before,after,check_exact=False,rtol=1e-9,atol=1e-7)
        numeric = before.select_dtypes(include='number').columns
        differences = (before[numeric]-after[numeric]).abs()
        status.update(status='replay_pass', rows=len(after), models=after.model.nunique(),
                      targets_per_model=6048, folds=after.key.nunique(),
                      max_absolute_numeric_difference=float(differences.max().max()),
                      historical_manuscript_guard='not executed; see audited source transform',
                      historical_manuscript_sha256=json.loads((archive_trial/'protocol.json').read_text())['protected_manuscript_sha256'])
        print('PASS: fitted-model replay matches all archived primary pathways.')
        verify(root,output)
        status['status'] = 'pass'
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'portability_audit.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


def train(root,output,limit):
    root, output = prepare_output(root, output)
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
