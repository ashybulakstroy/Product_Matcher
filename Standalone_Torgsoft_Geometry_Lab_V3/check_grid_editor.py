"""Browser regression tests. Run using the existing project Playwright env."""
from pathlib import Path
import json,zipfile,subprocess
from playwright.sync_api import sync_playwright
from geometry_lab.linked_grid import validate_linked_annotation

ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'experiment_outputs/standalone_geometry_v3_stage4';QA=OUT/'editor_qa'


def main():
    QA.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True);page=browser.new_page(viewport={'width':1600,'height':1100},accept_downloads=True)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((OUT/'editor.html').as_uri());page.wait_for_function('im.complete && im.naturalWidth>100')
        before=page.evaluate('JSON.parse(JSON.stringify(doc().raw))')
        with page.expect_download() as d:page.locator('#save').click()
        baseline=QA/'unchanged.json';d.value.save_as(baseline);saved=json.loads(baseline.read_text())
        assert saved['horizontal_polylines']==before['horizontal_polylines'] and saved['vertical_polylines']==before['vertical_polylines']
        if 'nodes'in before:assert saved['nodes']==before['nodes']
        assert saved['human_verified']==before['human_verified']
        def click_source(x,y,double=False):
            page.locator('#viewport').evaluate('(v,p)=>{let z=+$embed.value/100;v.scrollTop=Math.max(0,p.y*z-v.clientHeight/2);v.scrollLeft=Math.max(0,p.x*z-v.clientWidth/2)}'.replace('$embed','document.getElementById("zoom")'),dict(x=x,y=y))
            xy=page.evaluate('(p)=>{let r=C.getBoundingClientRect();return{x:r.x+p.x*r.width/C.width,y:r.y+p.y*r.height/C.height}}',dict(x=x,y=y))
            (page.mouse.dblclick if double else page.mouse.click)(xy['x'],xy['y'])
        n=page.evaluate('doc().horizontal.length');page.locator('#draw').click()
        for y in [600,620,640]:click_source(300,y)
        assert page.evaluate('drawing && doc().horizontal.length')==n+3
        page.locator('#axis').select_option('vertical');nv=page.evaluate('doc().vertical.length');click_source(320,620);click_source(350,620)
        assert page.evaluate('drawing && doc().vertical.length')==nv+2
        page.keyboard.press('Escape');assert not page.evaluate('drawing')
        page.keyboard.press('Control+z');assert page.evaluate('doc().vertical.length')==nv+1
        page.locator('#undo').click();assert page.evaluate('doc().vertical.length')==nv
        for _ in range(3):page.locator('#undo').click()
        assert page.evaluate('doc().horizontal.length')==n
        # Shared source node moves by source units at two display scales.
        for zoom in [50,125]:
            page.locator('#zoom').fill(str(zoom));page.locator('#zoom').dispatch_event('input')
            pt=page.evaluate('(()=>{let id=doc().horizontal[1].find(n=>doc().vertical.some(v=>v.includes(n)));return{id,...doc().points[id]}})()')
            page.locator('#viewport').evaluate('(v,p)=>{v.scrollTop=Math.max(0,p.y*+$embed.value/100-v.clientHeight/2)}'.replace('$embed','document.getElementById("zoom")'),pt)
            xy=page.evaluate('(p)=>{let r=C.getBoundingClientRect();return{x:r.x+p.x*r.width/C.width,y:r.y+p.y*r.height/C.height}}',pt)
            page.mouse.move(xy['x'],xy['y']);page.mouse.down();page.mouse.move(xy['x']+10*zoom/100,xy['y']+6*zoom/100,steps=4);page.mouse.up()
            moved=page.evaluate('(id)=>doc().points[id]',pt['id']);assert abs(moved['x']-pt['x']-10)<.01 and abs(moved['y']-pt['y']-6)<.01
            page.locator('#undo').click()
        page.locator('#zoom').fill('75');page.locator('#zoom').dispatch_event('input')
        # A bend is retained in polylines, not discarded by matrix conversion.
        segment=page.evaluate('(()=>{let h=doc().horizontal[2],d=doc();return {x:(d.points[h[1]].x+d.points[h[2]].x)/2,y:(d.points[h[1]].y+d.points[h[2]].y)/2}})()')
        old_count=page.evaluate('doc().horizontal[2].length');click_source(segment['x'],segment['y'],True)
        assert page.evaluate('doc().horizontal[2].length')==old_count+1
        with page.expect_download() as d:page.locator('#save').click()
        edited=QA/'edited.json';d.value.save_as(edited);changed=json.loads(edited.read_text());assert changed['human_verified'] is False
        validate_linked_annotation(changed)
        lab=Path(__file__).resolve().parent
        subprocess.run([str(lab/'.venv/Scripts/python.exe'),'-c',
            'import json,sys;from geometry_lab.structure_diagnostics import reference_data;g=json.load(open(sys.argv[1]));reference_data(g,g["source_shape"])',str(edited)],cwd=lab,check=True)
        page.locator('#open').set_input_files(str(edited));page.wait_for_function('doc().raw.schema==="geometry_grid_graph_v1"')
        with page.expect_download() as d:page.locator('#save').click()
        reopened=QA/'reopened.json';d.value.save_as(reopened);again=json.loads(reopened.read_text());assert again['horizontal_polylines']==changed['horizontal_polylines'] and again['vertical_polylines']==changed['vertical_polylines']
        with page.expect_download() as d:page.locator('#all').click()
        archive=QA/'all.zip';d.value.save_as(archive)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;manifest=json.loads(z.read('manifest.json'));assert len(manifest['files'])==18
            assert len(z.namelist())==19
        page.screenshot(path=str(OUT/'editor_preview.png'))
        page.goto((OUT/'final_control_editor.html').as_uri());page.wait_for_function('im.complete && im.naturalWidth>100')
        with page.expect_download() as d:page.locator('#save').click()
        empty=QA/'empty_unverified.json';d.value.save_as(empty);assert json.loads(empty.read_text())['human_verified'] is False
        assert not page.locator('input[type=checkbox]').count();assert not errors,errors;browser.close()
    (OUT/'editor_checks.json').write_text(json.dumps(dict(continuous_horizontal_and_vertical=True,shared_node_drag_at_two_scales=True,undo=True,bend_node=True,one_click_json=True,empty_download=True,zip_crc_and_manifest=True,reimport=True,coordinates_unchanged_on_save=True,no_auto_confirmation=True),indent=2))
    print('Editor regression checks PASS')


if __name__=='__main__':main()
