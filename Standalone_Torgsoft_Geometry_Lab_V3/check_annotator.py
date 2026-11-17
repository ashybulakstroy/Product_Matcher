from pathlib import Path
import json
from playwright.sync_api import sync_playwright
R=Path.cwd();lab=R/'Standalone_Torgsoft_Geometry_Lab_V3';out=R/'experiment_outputs/standalone_geometry_v3'
with sync_playwright() as pw:
 browser=pw.chromium.launch(channel='msedge',headless=True);page=browser.new_page(viewport={'width':1500,'height':1400})
 for f in lab.joinpath('ground_truth').glob('*.json'):
  data=json.loads(f.read_text(encoding='utf-8'))
  if data.get('annotation_origin')!='agent_visual_source_only':continue
  page.goto((lab/'annotate.html').as_uri())
  page.locator('#photo').set_input_files(str(R/'web_outputs'/data['job_id']/'original.png'))
  page.wait_for_function('document.getElementById("canvas").height===1100')
  page.locator('#sha').fill(data['sha256']);canvas=page.locator('#canvas')
  size=page.evaluate('({w:document.getElementById("canvas").width,h:document.getElementById("canvas").height})')
  height,width=data['source_shape'];scale=min(1,1200/width,1100/height)
  for axis,key in [('vertical','vertical_polylines'),('horizontal','horizontal_polylines')]:
   page.locator('#axis').select_option(axis)
   for line in data[key]:
    for x,y in line:
     canvas.click(position={'x':x*scale,'y':y*scale})
    page.locator('#end').click()
  with page.expect_download() as d:page.locator('#save').click()
  temp=out/'annotation_download.json';d.value.save_as(temp)
  actual=json.loads(temp.read_text(encoding='utf-8'))
  assert len(actual['vertical_polylines'])==5
  assert len(actual['horizontal_polylines'])==len(data['horizontal_polylines'])
  actual.update({k:v for k,v in data.items() if k not in ['vertical_polylines','horizontal_polylines']})
  actual['annotation_type']='agent_visual_source_only'
  f.write_text(json.dumps(actual,ensure_ascii=False,indent=2),encoding='utf-8')
  page.screenshot(path=str(out/(data['job_id'][:10]+'_annotation.png')))
 browser.close()
print('5 visual reference sets entered and exported through offline annotator')
