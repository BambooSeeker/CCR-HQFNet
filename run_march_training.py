"""Refit all March routing pathways using the released fixed carrier outputs."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/march_training')
    args = parser.parse_args()
    data = ROOT/'demo_data/march_training'
    for name, expected in json.loads((data/'checksums.json').read_text()).items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest() == expected, name
    fold = json.loads((data/'fold.json').read_text())[0]
    frame = pd.read_csv(data/'inputs.csv')
    assert frame.time.is_unique
    for key, count in [('train',4320),('val',336),('test',672)]:
        assert frame.delivery_day.isin(fold[key]).sum() == count, key
    assert len(frame)==5328
    env = os.environ.copy()
    env.update(USE_TF='0', TRANSFORMERS_NO_TF='1', MPLBACKEND='Agg')
    command=[sys.executable,str(ROOT/'code/experiment_pipeline/run_r10_joint_route_fold_carrier.py')]
    for flag,file in [('data','inputs.csv'),('labels','labels.csv'),('manifest','features.json'),('folds','fold.json'),('config','route_config.json'),('carrier-cache','carrier_cache.csv')]:
        command += ['--'+flag,str(data/file)]
    command += ['--output',str(args.output)]
    start=time.monotonic()
    subprocess.run(command,check=True,env=env,cwd=ROOT)
    new=pd.read_csv(args.output/'predictions.csv')
    ref=pd.read_csv(data/'reference_pathways.csv')
    merged=new.merge(ref,on=['time','mode'],validate='one_to_one',suffixes=('_new','_reference'))
    assert len(new)==len(ref)==len(merged)==4032
    delta=float(abs(merged.predicted_new-merged.predicted_reference).max())
    assert delta < 1e-8, f'Prediction mismatch: {delta}'
    full=new[new['mode']=='energy_consistent_dual_route']
    mae=float(abs(full.actual-full.predicted).mean())
    report={'scope':'Routing refit conditional on fixed Chronos-2 outputs','training_rows':4320,'validation_rows':336,'test_rows':672,'pathways':6,'maximum_prediction_difference':delta,'full_mae':mae,'elapsed_seconds':time.monotonic()-start,'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','lightgbm']}}
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('PASS: fitted March pathways reproduce the archived predictions.')

if __name__=='__main__':
    main()
