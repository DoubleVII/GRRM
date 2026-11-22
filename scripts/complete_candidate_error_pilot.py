"""Finish frozen pilot: shared location graphs, dev selection, calibration and holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from eval.candidate_error_core import aggregate, best_threshold, category_mapping, curve
from data.merge_run_oss_candidate_error_matching import prepare_rows
from eval.run_candidate_error_eval import report_rows
from eval.candidate_error_contrast import evaluate_contrast
from scripts.run_candidate_error_pilot import run, OSS, ENDPOINTS
from scripts.run_candidate_error_baseline import run as baseline
from scripts.audit_candidate_error_pilot import audit
from utils.candidate_error_protocol import read_predictions, write_json, digest
from utils.jsonl import read_jsonl

SCORING={'severity':'severity','fixed_probability':'probability','fixed_categorical':'categorical',
         'joint_probability':'probability','joint_categorical':'categorical','dual_probability':'probability',
         'candidate_free':'probability'}


def collect(root,backend,phase):
    directory=root/'predictions'/backend/(phase+'_v2')
    completed=json.loads((directory/'completed.json').read_text())
    return {name:read_predictions(directory/f'{name}.jsonl') for name in completed['variants']}


def shared_graphs(predictions,silver,output, *, model_path=OSS, base_url=None, workers=32, rescue_max_tokens=None):
    """One judge request per candidate for the union of location proposals across variants."""
    union={}
    for values in predictions.values():
        for r in values:
            u=union.setdefault(r['id'],dict(r,result={'spans':[]},status='ok'))
            for s in (r.get('result') or {}).get('spans',[]):
                if not any((s['source_start'],s['source_end'])==(p['source_start'],p['source_end']) for p in u['result']['spans']):
                    u['result']['spans'].append(s)
    rows,audit_info=prepare_rows(list(union.values()),silver,output,model_path=model_path,
        base_url=base_url or ENDPOINTS['oss'],workers=workers,temperature=0.,max_tokens=2048,retry=2,
        rescue_max_tokens=rescue_max_tokens)
    graphs={r['id']:r for r in rows}
    variants={}
    for name,values in predictions.items():
        selected=[]
        for r in values:
            if r['id'] not in graphs:
                continue
            graph=graphs[r['id']]
            spans=(r.get('result') or {}).get('spans',[])
            by_loc={(s['source_start'],s['source_end']):i for i,s in enumerate(graph['predicted'])}
            mapped={i:by_loc[s['source_start'],s['source_end']] for i,s in enumerate(spans)}
            edges=[[i,b] for i,j in mapped.items() for a,b in graph['edges'] if a==j]
            selected.append(dict(graph,predicted=spans,edges=edges,prediction_status=r['status']))
        variants[name]=selected
    audit_info.pop('prediction_failed',None)
    audit_info['synthetic_union_graph']=True
    audit_info['raw_prediction_failures_by_variant']={name:sum(r['status']!='ok' for r in values)
                                                   for name,values in predictions.items()}
    write_json(output/'audit.json',audit_info)
    return variants


def token_cost(records):
    return sum(a.get('response',{}).get('usage',{}).get('completion_tokens',0)
               for r in records for a in r.get('generation',{}).get('attempts',[]))/max(1,len(records))


def development(root,silver):
    selection=dict(dual_selected={},default_variant={},criteria='Development natural F1; dual requires +.02 F1, >= validity, <=1.5x output tokens',
                   silver_digest=digest(silver),statistics={})
    for backend in ('inst','oss'):
        predictions=collect(root,backend,'dev')
        variants=shared_graphs(predictions,silver,root/'evaluation'/backend/'dev_graph')
        summaries={}
        for name,rows in variants.items():
            natural=[r for r in rows if r['item']['cohort']=='natural']
            scoring=SCORING[name]
            # Development thresholds are diagnostics only, never deployed to test.
            mapping=category_mapping(natural) if scoring=='categorical' else None
            threshold=best_threshold(natural,scoring,mapping)
            report=report_rows(natural,scoring,threshold,mapping)
            summaries[name]=dict(threshold=threshold,mapping=mapping,metrics=report['overall'],
                                 mean_output_tokens=token_cost(predictions[name]))
            write_json(root/'evaluation'/backend/'dev'/f'{name}.json',dict(report=report,rows=rows))
        selection['dual_selected'][backend] = False
        if 'dual_probability' in summaries:
            source, dual = summaries['joint_probability'], summaries['dual_probability']
            sm, dm = source['metrics'], dual['metrics']
            selection['dual_selected'][backend] = bool((dm['f1'] or 0) >= (sm['f1'] or 0) + .02 and
                dm['prediction_valid'] >= sm['prediction_valid'] and
                dual['mean_output_tokens'] <= 1.5 * source['mean_output_tokens'])
        names = [name for name in summaries if name != 'dual_probability' or selection['dual_selected'][backend]]
        selection['default_variant'][backend]=max(names,key=lambda name:(summaries[name]['metrics']['f1'] or 0.,
                                                          -summaries[name]['metrics']['triggers_per_100']))
        selection['statistics'][backend]=summaries
    write_json(root/'development_selection.json',selection)
    return selection


def heldout_reports(root,silver):
    predictions_all={};settings_all={}
    for backend in ('inst','oss'):
        predictions=collect(root,backend,'heldout')
        sourcefree=root/'baseline'/f'{backend}.copied.jsonl'
        predictions['candidate_free']=[r for r in read_jsonl(sourcefree) if r['item']['split']!='dev']
        variants=shared_graphs(predictions,silver,root/'evaluation'/backend/'heldout_graph')
        for name,rows in variants.items():
            scoring=SCORING[name]
            cal=[r for r in rows if r['item']['split']=='calibration' and r['item']['cohort']=='natural']
            mapping=category_mapping(cal) if scoring=='categorical' else None
            threshold=best_threshold(cal,scoring,mapping)
            points=curve(cal,scoring,mapping)
            budgets={}
            for budget in (10,25,50,100):
                affordable=[p for p in points if p['triggers_per_100']<=budget]
                budgets[str(budget)]=max(affordable,key=lambda p:(p['recall'] or 0.,-p['triggers_per_100']))['threshold']
            cfg=dict(scoring=scoring,threshold=threshold,mapping=mapping,budget_thresholds=budgets,
                     calibration_ids=[r['id'] for r in cal],calibration_digest=digest(cal))
            reports={}
            for split in ('calibration','test'):
                for cohort in ('natural','contrast'):
                    chosen=[r for r in rows if r['item']['split']==split and r['item']['cohort']==cohort]
                    if not chosen:
                        continue
                    report=report_rows(chosen,scoring,threshold,mapping)
                    report['calibrated_budget_points']={b:dict(threshold=t,**aggregate(chosen,t,scoring,mapping)) for b,t in budgets.items()}
                    m=report['overall']
                    report['major_penalty_sensitivity']={str(w):m['fp']+w*m['fn'] for w in (1,2,5)}
                    reports[f'{split}.{cohort}']=report
            write_json(root/'evaluation'/backend/'heldout'/f'{name}.json',dict(config=cfg,reports=reports,rows=rows))
            predictions_all[f'{backend}.{name}']=predictions[name]
            settings_all[f'{backend}.{name}']=cfg
    # Only held-out candidates are supplied; independent confirmation is based on silver.
    evaluate_contrast([r for r in silver if r['item']['split']=='test'],predictions_all,settings_all,
                       root/'contrast',OSS,base_url=ENDPOINTS['oss'])


def main(root):
    root=Path(root)
    while not ((root/'silver/final.jsonl').exists() and all((root/'predictions'/b/'dev_v2/completed.json').exists() for b in ('inst','oss'))):
        print('Waiting for silver and completed development predictions',flush=True)
        time.sleep(30)
    silver=read_jsonl(root/'silver/final.jsonl')
    if not (root/'development_selection.json').exists():
        development(root,silver)
    else:
        frozen=json.loads((root/'development_selection.json').read_text())
        if frozen['silver_digest']!=digest(silver):
            raise ValueError('Silver changed after development selection; use a new experiment directory')
    print('Development configuration frozen',flush=True)
    if not (root/'secondary_analysis_frozen.json').exists():
        write_json(root/'secondary_analysis_frozen.json',dict(
            policy='Retain predicted severity=major before confidence thresholding; calibrate on natural calibration only.',
            reason='Supplementary severity/confidence interaction diagnostic',
            heldout_reports_existed_at_freeze=False,primary_protocol_unchanged=True,additional_model_calls=0))
    def finish_backend(backend):
        if (root/f'{backend}.prediction_protocol_frozen.json').exists():
            # Avoid two writers when optional blinded precomputation is still active.
            while not ((root/'predictions'/backend/'heldout_v2/completed.json').exists() and
                       (root/'baseline'/f'{backend}.copied.jsonl').exists()):
                time.sleep(15)
            return
        run(root,'heldout',backend)
        baseline(root,backend)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(finish_backend,('inst','oss')))
    heldout_reports(root,silver)
    audit(root)
    write_json(root/'completed.json',dict(status='complete',silver_digest=digest(silver)))
    print('PILOT COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/candidate_error_pilot_20261114')
    main(p.parse_args().root)
