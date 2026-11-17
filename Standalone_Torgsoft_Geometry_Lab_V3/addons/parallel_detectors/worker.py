"""Each algorithm executes in its OWN process; the parent enforces timeouts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
from detectors import detect, geometry_current, GEOMETRY_METHODS, DependencyMissing, AlgorithmUnavailable
from metrics import draw_lines, compare


def read_image(path):
    raw=np.frombuffer(Path(path).read_bytes(),dtype=np.uint8)
    img=cv2.imdecode(raw,cv2.IMREAD_COLOR)
    if img is None:raise ValueError('unreadable image')
    return img


def load_gt(path):
    if not path:return None
    p=Path(path)
    if not p.is_file():return None
    try:
        d=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(d,dict):raise ValueError('GT must be object')
        return d
    except (OSError,TypeError,ValueError) as exc:
        return {'_invalid':str(exc)}


def perform(args):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if getattr(args, 'start_gate', None):
        from datetime import datetime, timezone
        Path(args.ready_file).write_text(json.dumps(dict(pid=os.getpid(), ready_at_utc=datetime.now(timezone.utc).isoformat())), encoding='utf-8')
        barrier_start=time.monotonic()
        while not Path(args.start_gate).exists():
            if time.monotonic()-barrier_start>125:
                raise TimeoutError('Parent did not release the start barrier')
            time.sleep(.01)
    start=time.perf_counter()
    from datetime import datetime, timezone
    row={'method':args.method,'job_id':args.job_id,'sha256':args.sha,'status':'ERROR'}
    row['detector_started_at_utc']=datetime.now(timezone.utc).isoformat()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    try:
        image=read_image(args.image)
        h,w=image.shape[:2]
        factor=min(1.,args.proc_max_side/max(h,w)) if args.proc_max_side else 1.
        if factor<1:
            process=cv2.resize(image,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_AREA)
        else:process=image
        sy=process.shape[0]/h;sx=process.shape[1]/w
        row.update(source_shape=list(image.shape), processed_shape=list(process.shape), sx=sx, sy=sy,
                   scale_mode='native' if not args.proc_max_side else f'max_side_{args.proc_max_side}')
        if args.method in GEOMETRY_METHODS:
            # Baseline maps full-resolution source; other methods use processed pixels.
            geometry_input=image if args.method=='baseline' else process
            found,info,rectified=geometry_current(args.method,geometry_input,Path(args.lab),
                                               Path(args.job_dir),args.job_id,args.sha)
            if rectified is not None:
                cv2.imwrite(str(out/'rectified.png'),rectified)
        else:
            found,info=detect(args.method,process,Path(args.lab))
        # Inverse resize pixel-centre mapping to *oriented original* pixels.
        if (sx != 1 or sy != 1) and args.method != 'baseline':
            for item in found:
                for point in item['points']:
                    point[0]=(point[0]+.5)/sx-.5
                    point[1]=(point[1]+.5)/sy-.5
        gt=load_gt(args.gt)
        if gt is not None and 'sha256' in gt and gt['sha256']!=args.sha:
            gt={'_invalid':'sha256 mismatch'}
        row.update(status=('NOT_AVAILABLE' if args.method=='baseline' else 'REJECTED')
                   if not found and args.method in GEOMETRY_METHODS
                   else ('NO_DETECTION' if not found else 'COMPLETED'),
                   observation_type='projected_geometry_grid' if args.method in GEOMETRY_METHODS else 'detected_physical_candidates',
                   observed_h=sum(p['axis']=='h' for p in found),
                   observed_v=sum(p['axis']=='v' for p in found),
                   detector_info=info,
                   gt=compare(gt,found,image.shape))
        overlay=draw_lines(image,found)
        if gt and '_invalid' not in gt:
            ground=[{'points':p} for k in ('horizontal_polylines','vertical_polylines') for p in gt.get(k,[])]
            overlay=draw_lines(overlay,ground,color=(0,185,0),thickness=2)
        path=out/'overlay.png'
        if not cv2.imwrite(str(path),overlay):raise OSError('Cannot write overlay')
        (out/'observed_lines.json').write_text(json.dumps({'sha256':args.sha,'method':args.method,
               'coordinate_system':'oriented_original_px','lines':found,'metadata':info},ensure_ascii=False,indent=2),encoding='utf-8')
    except AlgorithmUnavailable as err:
        row.update(status='ALGORITHM_UNAVAILABLE',reason=str(err))
    except DependencyMissing as err:
        row.update(status='SKIPPED_DEPENDENCY',reason=str(err))
    except Exception as err:
        row.update(status='ERROR',reason=f'{type(err).__name__}: {err}',traceback=traceback.format_exc(limit=3))
    row['duration_sec']=round(time.perf_counter()-start,3)
    row['detector_finished_at_utc']=datetime.now(timezone.utc).isoformat()
    (out/'result.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--lab',required=True)
    p.add_argument('--job-id',required=True)
    p.add_argument('--job-dir',required=True)
    p.add_argument('--sha',required=True)
    p.add_argument('--gt',default='')
    p.add_argument('--proc-max-side',type=int,default=1600)
    p.add_argument('--start-gate')
    p.add_argument('--ready-file')
    args=p.parse_args()
    perform(args)

if __name__=='__main__':main()
