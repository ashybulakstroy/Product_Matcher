from pathlib import Path
from playwright.sync_api import sync_playwright
P=Path('experiment_outputs/standalone_geometry_v3')
with sync_playwright() as pw:
 b=pw.chromium.launch(channel='msedge',headless=True);page=b.new_page(viewport={'width':1600,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((P/'summary.html').resolve().as_uri());page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)');page.screenshot(path=str(P/'summary_preview.png'))
 for path in sorted((P/'real_visual_references/cases').glob('*/case.html')):
  page.goto(path.resolve().as_uri());page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
 assert not errors,errors
 print('PASS: summary and all 10 comparison pages load images; no JS errors');b.close()
