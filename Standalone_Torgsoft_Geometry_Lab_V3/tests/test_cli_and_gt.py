import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from geometry_lab.synthetic import receipt_grid,deform
from geometry_lab.algorithms import detect_and_run
from geometry_lab.metrics import evaluate_ground_truth

PROJECT=Path(__file__).resolve().parents[1]


def test_cli_full_benchmark_synthetic_is_isolated(tmp_path):
    src=tmp_path/'web_outputs'/'synthetic_job'
    src.mkdir(parents=True)
    image,*_=receipt_grid()
    cv2.imwrite(str(src/'original.png'),image)
    before=(src/'original.png').read_bytes()
    output=tmp_path/'data'/'output'/'geometry_benchmark'
    cmd=[sys.executable,str(PROJECT/'run.py'),'benchmark',
         '--source',str(src.parent),'--output',str(output),
         '--run-id','test_case','--methods','baseline,perspective,grid_mesh']
    p=subprocess.run(cmd,check=False,text=True,capture_output=True,timeout=60)
    assert p.returncode==0,(p.stderr,p.stdout)
    assert (src/'original.png').read_bytes()==before
    run=output/'test_case'
    assert (run/'summary.html').is_file()
    assert (run/'run.json').is_file()
    records=json.loads((run/'run.json').read_text(encoding='utf-8'))['records']
    assert len(records)==3
    assert records[0]['method']=='baseline' and records[0]['status']=='NOT_AVAILABLE'
    assert records[1]['method']=='perspective' and records[1]['status'] in ('SUCCESS','GEOMETRY_WARNING')
    assert records[2]['method']=='grid_mesh' and records[2]['status'] in ('SUCCESS','GEOMETRY_WARNING')
    assert records[2]['ground_truth']['status']=='GT_MISSING'
    for record in records[1:]:
        assert (run/record['overlay']).is_file()


def test_output_inside_source_rejected(tmp_path):
    src=tmp_path/'web_outputs';src.mkdir()
    output=src/'illegal'
    cmd=[sys.executable,str(PROJECT/'run.py'),'discover',
         '--source',str(src),'--output',str(output)]
    p=subprocess.run(cmd,check=False,text=True,capture_output=True,timeout=30)
    assert p.returncode==2
    assert 'Refusing to write' in p.stderr
    assert not output.exists()


def test_wave_photo_must_not_fabricate_success():
    img,*_=receipt_grid()
    warped=deform(img,'wave')
    result,diag=detect_and_run(warped,'grid_mesh')
    if result is None:
        assert diag['status']=='REJECTED'
    else:
        # A calculated mapping is NOT manually proven accurate.
        assert result.info['physical_grid']=='observed_lines_source'


def test_annotate_file_exists():
    assert (PROJECT/'annotate.html').is_file()
    assert (PROJECT/'AGENT_TASK.md').is_file()
