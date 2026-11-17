import os,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
from geometry_lab.detection import Rule,_choose_verticals,_choose_horizontals
from geometry_lab.algorithms import Result
from geometry_lab.metrics import safety,evaluate_ground_truth
from geometry_lab.synthetic import receipt_grid

ROOT=Path(__file__).resolve().parents[1]
def rules():
 return [Rule('v',np.array([float(x)]),float(x),.7,(100,800)) for x in [100,160,470,560,700]]

def test_disjoint_support_is_not_a_five_column_grid():
 vv=rules();vv[-1].domain=(850,990)
 assert _choose_verticals(vv,900,1000) is None

def test_text_underline_cannot_replace_full_table_rule():
 hh=[Rule('h',np.array([float(y)]),float(y),.3,(200,400)) for y in [200,400,600]]
 assert not _choose_horizontals(hh,rules(),(1000,900))

def identity_result():
 y,x=np.indices((100,200),dtype=np.float32)
 return Result('identity',np.zeros((100,200,3),np.uint8),x,y,[10,30,100,140,180],[10,50,90],{})

def test_fold_is_invalid_even_when_pixels_are_inside():
 r=identity_result();r.map_x=r.map_x[:,::-1].copy()
 assert safety(r,r.image.shape)['folded_fraction']==1

def test_nonfinite_visual_reference_is_not_a_measured_error():
 r=identity_result()
 gt=dict(vertical_polylines=[[[float('nan'),0],[0,1]]]*5,horizontal_polylines=[[[0,0],[1,1]]]*3)
 assert evaluate_ground_truth(r,gt,r.image.shape)['status']=='GT_INVALID'

def test_gt_missing_lines_has_no_fake_zero_error():
 r=identity_result();g=evaluate_ground_truth(r,dict(vertical_polylines=[],horizontal_polylines=[]),r.image.shape)
 assert g['status']=='GT_MISMATCH' and 'mean_px' not in g

def test_downscaled_cli_keeps_one_map_and_original_pixels(tmp_path):
 source=tmp_path/'sources'/'one';source.mkdir(parents=True)
 image=receipt_grid()[0];image=cv2.resize(image,None,fx=2,fy=2)
 cv2.imwrite(str(source/'original.png'),image)
 before=(source/'original.png').read_bytes()
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 out=tmp_path/'result'
 process=subprocess.run([sys.executable,str(ROOT/'run.py'),'benchmark','--source',str(source.parent),
     '--output',str(out),'--run-id','small','--methods','grid_mesh','--proc-max-side','1000','--workers','4'],capture_output=True,text=True,env=env)
 assert process.returncode==0,process.stderr
 case=out/'small/cases/0001_one'
 with np.load(case/'grid_mesh_map.npz') as data:
  expected=cv2.remap(image,data['map_x'],data['map_y'],cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
 assert np.array_equal(expected,cv2.imread(str(case/'grid_mesh_rectified.png')))
 assert (source/'original.png').read_bytes()==before
