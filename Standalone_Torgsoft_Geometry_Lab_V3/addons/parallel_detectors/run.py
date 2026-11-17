#!/usr/bin/env python
"""Parallel, isolated add-on for Standalone Torgsoft Geometry Lab V3.

Runs all selected methods simultaneously per image (default ten worker slots).
Independent of older experiments and outputs. Does not modify main V3 code.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote

METHODS=('baseline','perspective','grid_mesh','tps_scipy','physical_grid_current','opencv_lsd','edge_drawing','img2table','frangi','sato')
THIS=Path(__file__).resolve().parent

def find_lab(arg):
    if arg is not None:
        lab=Path(arg).resolve()
    else:
        # Installed under V3/addons/parallel_detectors/
        lab=THIS.parent.parent
    if not (lab/'geometry_lab'/'detection.py').is_file():
        raise ValueError(f'V3 lab not found at {lab}; pass --lab')
    return lab


def scan(source, max_cases=10):
    source=Path(source).resolve()
    if not source.is_dir():raise FileNotFoundError(f'Missing source: {source}')
    seen=set();cases=[]
    names=('original.png','original.jpg','original.jpeg','upload.bin')
    for folder in sorted(source.iterdir()):
        if folder.is_symlink() or not folder.is_dir():continue
        file=next((folder/x for x in names if (folder/x).is_file() and not (folder/x).is_symlink()),None)
        if file is None:continue
        identity=folder/'upload.bin'
        if not identity.is_file() or identity.is_symlink():identity=file
        sha=hashlib.sha256(identity.read_bytes()).hexdigest()
        if sha in seen:continue
        seen.add(sha)
        cases.append({'id':folder.name,'sha':sha,'file':file})
        if max_cases and len(cases)>=max_cases:break
    return cases


def _safe_output(output, source, lab):
    src=Path(source).resolve()
    dest=Path(output).resolve()
    if dest==src or src in dest.parents:
        raise ValueError('Output inside web_outputs is prohibited')
    if dest==lab or dest==lab.parent:
        raise ValueError('Output must not overwrite the V3 lab or project root')
    return dest


def one_job(args, method, case, root, lab, gt_root):
    out=root/case['sha'][:16]/method
    out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(THIS/'worker.py'), '--image',str(case['file']),
         '--method',method,'--output',str(out),'--lab',str(lab),
         '--job-id',case['id'],'--job-dir',str(case['file'].parent),'--sha',case['sha'],'--proc-max-side',str(args.proc_max_side)]
    gt=gt_root/f"{case['sha']}.json"
    if gt.is_file():cmd.extend(['--gt',str(gt)])
    env=os.environ.copy()
    # Prevent 6 workers each internally using 10+ BLAS/OpenCV threads.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key]='1'
    return {'method':method,'case':case,'output':out,'command':cmd,'env':env}


def execute_parallel(jobs, workers=10, timeout_sec=120):
    """Start all methods immediately, up to `workers`, and enforce per-process timeout.

    Subprocess isolation works on Windows; a hung algorithm cannot block siblings.
    """
    waiting=list(jobs)
    active=[]
    done=[]
    worker_script=THIS/'worker.py'
    gate = jobs[0]['output'].parent / 'start_gate.json' if jobs else None
    released = False
    for job in jobs:
        job['ready_file'] = job['output'] / 'ready.json'
        job['command'] = list(job['command']) + ['--start-gate', str(gate), '--ready-file', str(job['ready_file'])]
    while waiting or active:
        while waiting and len(active)<workers:
            job=waiting.pop(0)
            start=time.monotonic()
            errfile=open(job['output']/'worker_stderr.log','wb')
            try:
                proc=subprocess.Popen(job['command'],stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL,stderr=errfile,env=job['env'],cwd=str(THIS))
            except OSError as exc:
                failed=dict(method=job['method'],status='ERROR',reason=f'Cannot start worker: {exc}',
                            job_id=job['case']['id'],sha256=job['case']['sha'],worker_pid=None,
                            started_at_utc=datetime.now(timezone.utc).isoformat(),
                            finished_at_utc=datetime.now(timezone.utc).isoformat())
                (job['output']/'result.json').write_text(json.dumps(failed),encoding='utf-8')
                done.append(failed)
                continue
            finally:
                errfile.close()
            active.append({'proc':proc,'start':start,'job':job,
                           'started_at_utc':datetime.now(timezone.utc).isoformat()})
        if not released and active and (not waiting or len(active)>=workers):
            ready = all(t['job']['ready_file'].exists() or t['proc'].poll() is not None for t in active)
            # A failed startup cannot prevent the other methods from running.
            if ready or max(time.monotonic()-t['start'] for t in active)>30:
                gate.write_text(json.dumps(dict(released_at_utc=datetime.now(timezone.utc).isoformat(),
                    all_ready=ready, ready_workers=sum(t['job']['ready_file'].exists() for t in active))),encoding='utf-8')
                released=True
        for task in list(active):
            proc=task['proc'];job=task['job']
            elapsed=time.monotonic()-task['start']
            exit_code=proc.poll()
            if exit_code is None and elapsed < timeout_sec:
                continue
            if exit_code is None:
                proc.kill(); proc.wait(); exit_code=proc.returncode; stderr=(job['output']/'worker_stderr.log').read_bytes()[-800:]
                result={'status':'TIMEOUT','reason':f'Worker exceeded {timeout_sec}s','method':job['method']}
            else:
                proc.wait(); stderr=(job['output']/'worker_stderr.log').read_bytes()[-800:]
                f=job['output']/'result.json'
                try:
                    result=json.loads(f.read_text(encoding='utf-8'))
                except (OSError,ValueError,TypeError):
                    result={'status':'ERROR','reason':f'Worker exited {exit_code} without valid result',
                            'method':job['method']}
            result.setdefault('method',job['method'])
            result.setdefault('duration_sec',round(elapsed,3))
            result['worker_exit_code']=exit_code
            result['worker_pid']=proc.pid
            result['started_at_utc']=task['started_at_utc']
            result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
            result['job_id']=job['case']['id']
            result['sha256']=job['case']['sha']
            result['stderr_tail']=stderr.decode('utf-8','replace')[-800:] if stderr else ''
            (job['output']/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            done.append(result)
            active.remove(task)
            print(f"{job['case']['id'][:14]:14s} {job['method']:22s} {result['status']:20s} {result['duration_sec']}s",flush=True)
        if active:time.sleep(.08)
    return done


def write_report(root,cases,results,meta):
    (root/'run.json').write_text(json.dumps({'metadata':meta,'records':results},ensure_ascii=False,indent=2),encoding='utf-8')
    chunks=['<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>Parallel grid detectors — V3 add-on</title>',
            '<style>body{font:15px system-ui;margin:26px;background:#f6f7f9;color:#15202b}h1{font-size:24px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#edf1f6}.case{margin-top:32px}section.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}article{background:white;padding:12px;border:1px solid #d9dfe8;border-radius:10px}img{width:100%;height:auto}small{color:#526071}a{color:#0b64b1}</style></head><body>',
            '<h1>Параллельный анализ печатной сетки · расширение V3</h1>',
            '<p>Каждый метод работает в отдельном процессе. Голубые линии — кандидаты детектора или проекция модельной сетки; зелёные — доступная ручная разметка. '
            'Обнаружение сегментов не доказывает правильность товарных строк. Отсутствие GT не приравнивается к 0% точности.</p>',
            '<p><b>Воркеров:</b> '+str(meta['workers'])+'; <b>методов:</b> '+str(len(meta['methods']))+'; '
            '<b>реальных/входных фото:</b> '+str(len(cases))+'</p>',
            '<table><tr><th>Метод</th><th>Обработано</th><th>Нет зависимостей</th><th>Нет линий</th><th>Ошибки/таймауты</th><th>Средняя длительность, с</th></tr>']
    for method in meta['methods']:
        subset=[r for r in results if r['method']==method]
        good=sum(x['status']=='COMPLETED' for x in subset)
        skip=sum(x['status']=='SKIPPED_DEPENDENCY' for x in subset)
        empty=sum(x['status']=='NO_DETECTION' for x in subset)
        bad=len(subset)-good-skip-empty
        durations=[r.get('duration_sec',0) for r in subset]
        chunks.append(f'<tr><td>{escape(method)}</td><td>{good}/{len(subset)}</td><td>{skip}</td><td>{empty}</td><td>{bad}</td><td>{sum(durations)/max(1,len(durations)):.3f}</td></tr>')
    chunks.append('</table>')
    for case in cases:
        sha=case['sha'];chunks.append(f'<h2 class="case">{escape(case["id"])} · {sha[:18]}…</h2><section class="grid">')
        for method in meta['methods']:
            r=next((x for x in results if x['method']==method and x['sha256']==sha),None)
            if not r:continue
            image=root/sha[:16]/method/'overlay.png'
            asset=image.relative_to(root).as_posix() if image.is_file() else None
            gt=r.get('gt',{})
            desc=[]
            if gt:
                desc.append('GT: '+str(gt.get('status')))
                for axis in ('h','v'):
                    if axis in gt:desc.append(axis+' recall='+str(gt[axis].get('recall_sample_pct'))+'%')
            status=escape(r['status'])
            rectified = root/sha[:16]/method/'rectified.png'
            rectified_tag = f'<img loading="lazy" src="{quote(rectified.relative_to(root).as_posix())}" alt="rectified">' if rectified.is_file() else ''
            img_tag=f'<img loading="lazy" src="{quote(asset)}" alt="{escape(method)} overlay">' if asset else ''
            observed = root/sha[:16]/method/'observed_lines.json'
            observed_link = f' | <a href="{sha[:16]}/{escape(method)}/observed_lines.json">Кандидаты линий</a>' if observed.is_file() else ''
            chunks.append(f'<article><h3>{escape(method)}</h3><p>{status}; {r.get("duration_sec",0)}s</p><small>{escape(" | ".join(desc))}</small>'
                          f'{img_tag}{rectified_tag}<p><a href="{sha[:16]}/{escape(method)}/result.json">JSON-отчёт</a>{observed_link}</p></article>')
        chunks.append('</section>')
    chunks.append('</body></html>')
    (root/'summary.html').write_text('\n'.join(chunks),encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description='All line detectors in parallel, without modifying V3')
    p.add_argument('--lab',type=Path,help='Path to Standalone_Torgsoft_Geometry_Lab_V3')
    p.add_argument('--source',type=Path,help='Project web_outputs; read-only')
    p.add_argument('--output',type=Path,help='Dedicated experiment output directory')
    p.add_argument('--ground-truth',type=Path,help='Ground truth directory from V3')
    p.add_argument('--methods',default=','.join(METHODS))
    p.add_argument('--workers',type=int,default=10,help='Simultaneous algorithm processes per photo (1..10)')
    p.add_argument('--timeout',type=int,default=120,help='Per algorithm per photo seconds (10..600)')
    p.add_argument('--max-cases',type=int,default=10,help='0=all unique uploads')
    p.add_argument('--proc-max-side',type=int,default=1500,help='Downscale for detection')
    p.add_argument('--run-id',default=None)
    args=p.parse_args(argv)
    if not 1<=args.workers<=10:raise ValueError('--workers must be 1..10')
    if not 10<=args.timeout<=600:raise ValueError('--timeout must be 10..600')
    if args.proc_max_side != 0 and not 450<=args.proc_max_side<=3500:raise ValueError('--proc-max-side must be 0 (native) or 450..3500')
    methods=[x.strip() for x in args.methods.split(',') if x.strip()]
    if len(set(methods))!=len(methods) or not methods or set(methods)-set(METHODS):
        raise ValueError('Invalid/duplicate methods: '+str(methods))
    lab=find_lab(args.lab)
    project=lab.parent
    source=args.source.resolve() if args.source else project/'web_outputs'
    output=args.output.resolve() if args.output else project/'experiment_outputs'/'standalone_geometry_v3'/'parallel_detectors'
    output=_safe_output(output,source,lab)
    gt=args.ground_truth.resolve() if args.ground_truth else lab/'ground_truth'
    cases=scan(source,args.max_cases)
    if not cases:raise RuntimeError('No input images found')
    run_id=args.run_id or datetime.now().strftime('lines_%Y%m%d_%H%M%S')
    if not re.fullmatch('[A-Za-z0-9_-]{1,90}',run_id):raise ValueError('Unsafe run-id')
    root=output/run_id
    if root.exists():raise FileExistsError(f'Output already exists: {root}')
    root.mkdir(parents=True)
    metadata={'lab':str(lab),'source':str(source),'output':str(root),'methods':methods,
              'workers':args.workers,'timeout_sec':args.timeout,'cases':len(cases),
              'execution':'concurrent_independent_subprocesses_per_case',
              'V3_modified':False,'working_OCR_modified':False,
              'warning':'COMPLETED does not prove complete physical row boundaries; model projection differs from observation'}
    all_results=[]
    for index,case in enumerate(cases,1):
        print(f'[{index}/{len(cases)}] {case["id"]} -> starting {len(methods)} algorithms concurrently',flush=True)
        jobs=[one_job(args,m,case,root,lab,gt) for m in methods]
        all_results.extend(execute_parallel(jobs,args.workers,args.timeout))
    write_report(root,cases,all_results,metadata)
    print('Report:',root/'summary.html')
    return root

if __name__=='__main__':main()
