#!/usr/bin/env python3
"""Standalone command line: run from ANY cwd. Python 3.10+.

No import from the working App. Reads from web_outputs without modification.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
import cv2
import numpy as np
from geometry_lab.cases import discover, load_image, load_baseline
from geometry_lab.algorithms import detect_and_run
from geometry_lab.metrics import safety,render_overlay,render_source_overlay,evaluate_ground_truth
from geometry_lab.reports import jdump,html_report

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
METHODS=('baseline','perspective','grid_mesh','tps_scipy')


def _paths(args):
    project=Path(args.project_root).resolve() if args.project_root else PROJECT
    source=Path(args.source).resolve() if args.source else project/'web_outputs'
    output=Path(args.output).resolve() if args.output else project/'data'/'output'/'geometry_benchmark'
    if output==source or source in output.parents:
        raise ValueError('Refusing to write output inside web_outputs (read-only)')
    if output in (HERE, PROJECT) or project==output:
        raise ValueError('Output must be dedicated data/output subdirectory')
    return source,output


def _encode(path,image):
    if not cv2.imwrite(str(path),image):raise IOError(f'Cannot write {path}')


def _downscale(image,limit):
    h,w=image.shape[:2]
    fac=min(1.,limit/max(h,w))
    if fac>=.999:return image,1.
    return cv2.resize(image,(round(w*fac),round(h*fac)),interpolation=cv2.INTER_AREA),fac


def _load_gt(gt_dir,sha):
    path=gt_dir/f'{sha}.json'
    if not path.is_file():return None
    try:
        d=json.loads(path.read_text(encoding='utf-8'))
        if d.get('sha256') and d['sha256']!=sha:raise ValueError('GT sha256 mismatch')
        return d
    except (ValueError,TypeError,json.JSONDecodeError) as error:
        return {'_invalid':str(error)}


def _run_case(payload):
    n,total,case,run_dir,gt_dir,proc_max_side,wanted=payload
    cv2.setNumThreads(1)
    records=[]
    case_name=re.sub('[^a-zA-Z0-9_-]','_',case.job_id)[:60]
    case_out=run_dir/'cases'/f'{n:04d}_{case_name}'
    case_out.mkdir(parents=True)
    try:
        original=load_image(case)
    except Exception as error:
        records.append({'job_id':case.job_id,'method':'source','status':'INPUT_ERROR','reason':str(error)})
        return records
    gt=_load_gt(gt_dir,case.sha256)
    reduced,scale=_downscale(original,proc_max_side)
    case_meta={'job_id':case.job_id,'sha256':case.sha256,'input_file':str(case.image_path),
               'source_shape':original.shape[:2],'processed_shape':reduced.shape[:2],
               'process_scale':round(scale,6),
               'gt_status':'GT_PRESENT' if gt and '_invalid' not in gt else ('GT_INVALID' if gt else 'GT_MISSING')}
    jdump(case_out/'case.json',case_meta)
    _encode(case_out/'original.png',original)
    for method in wanted:
        start=time.perf_counter()
        row={'job_id':case.job_id,'sha256':case.sha256,'method':method,'status':'NOT_RUN'}
        result=None
        try:
            if method=='baseline':
                result,reason=load_baseline(case)
                if result is None:
                    row.update(status='NOT_AVAILABLE',reason=reason)
            else:
                result,detector=detect_and_run(reduced,method)
                jdump(case_out/f'{method}_detector.json',detector)
                if result is None:
                    row.update(status='REJECTED',reason=detector.get('reason','unknown'))
                else:
                    # Source mapping must always refer to oriented, full-resolution original.
                    if scale!=1.:
                        result.map_x=((result.map_x+.5)/(reduced.shape[1]/original.shape[1])-.5).astype(np.float32)
                        result.map_y=((result.map_y+.5)/(reduced.shape[0]/original.shape[0])-.5).astype(np.float32)
                        result.image=cv2.remap(original,result.map_x,result.map_y,cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
                    result.info['processing_scale']=round(scale,6)
            if result is not None:
                valid=safety(result,original.shape)
                gt_result=({'status':'GT_MISSING'} if gt is None else
                           {'status':'GT_INVALID','reason':gt['_invalid']} if '_invalid' in gt else
                           evaluate_ground_truth(result,gt,original.shape))
                overlay,detected=render_overlay(result,original)
                stem=f'cases/{case_out.name}/{method}'
                _encode(case_out/f'{method}_rectified.png',result.image)
                np.savez_compressed(case_out/f'{method}_map.npz',map_x=result.map_x,map_y=result.map_y)
                jdump(case_out/f'{method}_grid.json',dict(xs=result.xs,ys=result.ys))
                _encode(case_out/f'{method}_overlay.png',overlay)
                _encode(case_out/f'{method}_original_overlay.png',render_source_overlay(result,original))
                row.update(status='SUCCESS' if valid.get('folded_fraction',1.)==0 and
                           valid.get('valid_fraction',0.)>.9 else 'GEOMETRY_WARNING',
                           metrics=valid,ground_truth=gt_result,
                           detector_repeat_count=detected,model=result.info,
                           rectified=stem+'_rectified.png',overlay=stem+'_overlay.png',
                           source_overlay=stem+'_original_overlay.png')
        except Exception as error:
            row.update(status='ERROR',reason=f'{type(error).__name__}: {error}')
        row['duration_sec']=round(time.perf_counter()-start,3)
        jdump(case_out/f'{method}.json',row)
        records.append(row)
        print(f'[{n}/{total}] {case.job_id[:10]} {method}: {row["status"]} {row["duration_sec"]}s',flush=True)
    return records


def benchmark(args):
    source,output=_paths(args)
    wanted=[x.strip() for x in args.methods.split(',') if x.strip()]
    unknown=set(wanted)-set(METHODS)
    if unknown:raise ValueError(f'Unknown methods: {sorted(unknown)}')
    cases=discover(source,0)
    if args.case_ids:
        ids=set(args.case_ids.split(','))
        cases=[case for case in cases if case.job_id in ids]
    if args.max_cases: cases=cases[:args.max_cases]
    if not cases:raise RuntimeError('No unique images found under web_outputs/*/original.png or upload.bin')
    name=args.run_id or 'standalone_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',name):raise ValueError('Invalid run-id')
    run_dir=output/name
    if run_dir.exists():raise FileExistsError(f'Refusing to overwrite previous run: {run_dir}')
    run_dir.mkdir(parents=True,exist_ok=False)
    gt_dir=Path(args.ground_truth).resolve() if args.ground_truth else HERE/'ground_truth'
    records=[]
    payloads=[(n,len(cases),case,run_dir,gt_dir,args.proc_max_side,wanted)
              for n,case in enumerate(cases,1)]
    if args.workers == 1:
        for payload in payloads: records.extend(_run_case(payload))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(_run_case,payloads): records.extend(rows)
    metadata={'version':'3.0.0','source':str(source),'output':str(run_dir),
              'run_id':name,'unique_cases':len(cases),'workers':args.workers,'methods':wanted,
              'ground_truth':str(gt_dir),'no_prior_trial_imports':True,
              'baseline_origin':'CURRENT MAIN PROJECT web_outputs only',
              'warning':'SUCCESS is not accuracy; manual GT is required to confirm alignment'}
    jdump(run_dir/'run.json',{'metadata':metadata,'records':records})
    html_report(run_dir,records,metadata)
    print(f'REPORT: {run_dir / "summary.html"}')
    return 0


def main():
    p=argparse.ArgumentParser(description='Standalone geometry benchmark; never modifies web_outputs')
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['discover','benchmark']:
        x=sub.add_parser(command)
        x.add_argument('--project-root',type=Path)
        x.add_argument('--source',type=Path,help='web_outputs root')
        x.add_argument('--output',type=Path,help='separate output root')
        x.add_argument('--case-ids',default='',help='Optional selected discovered job IDs')
        x.add_argument('--max-cases',type=int,default=10,help='0 = all')
        if command=='benchmark':
            x.add_argument('--workers',type=int,choices=range(1,9),default=4)
            x.add_argument('--run-id',type=str)
            x.add_argument('--methods',default=','.join(METHODS))
            x.add_argument('--proc-max-side',type=int,default=1800)
            x.add_argument('--ground-truth',type=Path)
    args=p.parse_args()
    try:
        if args.command=='discover':
            source,_=_paths(args)
            cases=discover(source,args.max_cases)
            print(json.dumps([{'job_id':x.job_id,'sha256':x.sha256,
                               'image':str(x.image_path)} for x in cases],ensure_ascii=False,indent=2))
            return 0
        if args.proc_max_side<500 or args.proc_max_side>6000:
            raise ValueError('--proc-max-side must be 500..6000')
        return benchmark(args)
    except (RuntimeError,ValueError,FileNotFoundError,FileExistsError) as error:
        print('ERROR:',error,file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
