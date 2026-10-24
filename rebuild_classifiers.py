"""Portable classifier-only rebuilding from the released chronological inputs.

Default (no fitting):
    python rebuild_classifiers.py --archive /path/to/research_archive --output /separate/check

Explicit future fitting, not required for saved-model replay:
    python rebuild_classifiers.py --archive /path/to/research_archive --output /separate/refit --fit-classifiers

Use --key to restrict either operation to one archived fold. Both operations
initialize r36 with an explicit project root and use the release's timestamp-lag
restoration, feature lists, classifier fitter and three chronological holdouts.
The label threshold remains conditional on the outer training block. Fitting
rebuilds only the supply and congestion classifiers, not residual forecasters,
the low-price expert, probability mappings, controller policies or the carrier.
It does not replace the released canonical model bundles or establish numerical
equivalence of a new fit. The default preflight never invokes a fitting method.
All generated files, including source-transform audits, go under --output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

sys.dont_write_bytecode = True

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from reproduce_revision import load, prepare_output, replace_once, setup


def initialize(root, audit):
    project = root/'CCR_HQFNet_评审修正_物理拥塞路由_20260721'
    source_dir = project/'03_results/r47_timestamp_lags_v1.0/finalization_source'
    def relocate(source, changes):
        before = ("WORKSPACE=Path(__file__).resolve().parent\n"
                  "TRACE=next(WORKSPACE.rglob('point_metric_traceability.csv'))\n"
                  "PROJECT=TRACE.parents[2]")
        after = ("WORKSPACE=Path(__file__).resolve().parent\n"
                 f"PROJECT=Path({str(project)!r})\n"
                 "TRACE=PROJECT/'03_results/r21_reviewer1_numerical_traceability_v1.0/point_metric_traceability.csv'")
        return replace_once(source, before, after, changes,
                            'initialize r36 from the explicit archive project, not recursive trace discovery')
    base = load('r36_canonical_validation_search',
                source_dir/'r36_canonical_validation_search.py', relocate, audit)
    _, _, restore, _ = setup(root, audit)
    # Execute only the archived classifier definition; do not import its training driver.
    env = restore.wrappers.__dict__
    env.update(base=base, lgb=lgb, np=np)
    fitter_path = source_dir/'r41_history_validation.py'
    restore.extract(fitter_path, {'fit_classifier'}, env)
    audit.append(dict(module='archived_classifier_fitter', path=str(fitter_path),
                      source_sha256=hashlib.sha256(fitter_path.read_bytes()).hexdigest(),
                      extracted_definitions=['fit_classifier', 'RestrictedClassifier', 'RestrictedBooster']))
    return base, restore, env['fit_classifier']


def chronological_blocks(frame, meta, train, val):
    yield 'outer', train, val, val | frame.delivery_day.isin(pd.to_datetime(meta['fold']['test'])).to_numpy()
    days = pd.DatetimeIndex(pd.to_datetime(meta['fold']['train']))
    hold = days[-45:]
    for i in range(3):
        block = hold[15*i:15*(i+1)]
        prefix = days[days < block[0]]
        fit = frame.delivery_day.isin(prefix[:-7]).to_numpy()
        early = frame.delivery_day.isin(prefix[-7:]).to_numpy()
        target = frame.delivery_day.isin(block).to_numpy()
        if (fit & target).any() or (early & target).any():
            raise ValueError('A chronological classifier holdout overlaps fitting data')
        yield f'oof_{i}', fit, early, target


def rebuild(root, output, fit_classifiers=False, key=None):
    root, output = prepare_output(root, output)
    audit = []
    report = dict(status='running', fit_requested=fit_classifiers, training_calls=0,
                  scope='supply/congestion classifiers only; outer carrier/scalers/label threshold shared',
                  archive=str(root), output=str(output), sources=audit, folds=[])
    try:
        base, restore, fitter = initialize(root, audit)
        jobs = base.definitions()
        if key:
            jobs = [job for job in jobs if job['key'] == key]
            if not jobs:
                raise ValueError(f'Unknown fold key: {key}')
        for spec in jobs:
            folder = restore.R/'folds'/spec['key']
            meta = json.loads((folder/'validation_selected.json').read_text(encoding='utf-8'))
            frame, market, congestion, train, val, test = restore.read_fold(meta)
            y = frame.price_real.to_numpy(float)
            low = float(np.quantile(y[train & (y >= 0)], .1))
            supply = np.full(len(y), 2)
            supply[(y >= 0) & (y <= low)] = 1
            supply[y < 0] = 0
            congestion_labels = frame.congestion_state.fillna(-1).to_numpy(int)
            rec = dict(key=spec['key'], training_rows=int(train.sum()),
                       validation_rows=int(val.sum()), test_rows=int(test.sum()),
                       outer_low_threshold=low, blocks=[])
            for name, fit, early, target in chronological_blocks(frame, meta, train, val):
                block = dict(name=name, fitting_days=int(frame.loc[fit, 'delivery_day'].nunique()),
                             early_stopping_days=int(frame.loc[early, 'delivery_day'].nunique()),
                             prediction_days=int(frame.loc[target, 'delivery_day'].nunique()), routers=[])
                models = {}
                for router, cols, labels in [('supply', market, supply),
                                             ('congestion', market+congestion, congestion_labels)]:
                    valid = labels >= 0
                    fit_mask, early_mask = fit & valid, early & valid
                    fields = [col for col in cols if not col.startswith('carrier_')]
                    classes = np.unique(labels[fit_mask]).tolist()
                    if classes != [0, 1, 2]:
                        raise ValueError(f'{spec["key"]}/{name}/{router}: missing fitting classes {classes}')
                    detail = dict(router=router, features=fields, classes=classes,
                                  fitting_rows=int(fit_mask.sum()), early_stopping_rows=int(early_mask.sum()))
                    if fit_classifiers:
                        model = fitter(frame.loc[fit_mask, cols].to_numpy(float), labels[fit_mask],
                                       frame.loc[early_mask, cols].to_numpy(float), labels[early_mask], cols)
                        report['training_calls'] += 1
                        models[router+'_classifier'] = model
                        probabilities = model.predict_proba(frame.loc[target, cols].to_numpy(float))
                        detail['probability_rows'] = len(probabilities)
                    block['routers'].append(detail)
                if fit_classifiers:
                    dest = output/'folds'/spec['key']/name
                    dest.mkdir(parents=True, exist_ok=True)
                    joblib.dump(models, dest/'classifiers.joblib', compress=3)
                rec['blocks'].append(block)
            report['folds'].append(rec)
            print(json.dumps(dict(phase='classifier_fit' if fit_classifiers else 'classifier_preflight',
                                  key=spec['key'], blocks=len(rec['blocks']))), flush=True)
        report.update(status='pass', fold_count=len(report['folds']),
                      classifier_plan_count=sum(len(block['routers']) for fold in report['folds'] for block in fold['blocks']),
                      fitting_verified=False if not fit_classifiers else True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'classifier_rebuild_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key')
    parser.add_argument('--fit-classifiers', action='store_true', help='Explicitly fit classifiers; omitted for a no-training preflight')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    rebuild(args.archive, args.output, args.fit_classifiers, args.key)


if __name__ == '__main__':
    main()
