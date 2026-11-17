"""Build one-click editors without altering existing annotations."""
import json
from stage4 import ROOT,LAB,OUT,cases
from editor_documents import render_editor


def main():
    manifest=[]
    for c in cases():
        raw=json.loads((LAB/c['labels']/f"{c['sha256']}.json").read_text(encoding='utf-8'))
        manifest.append(dict(job=c['job'],sha256=c['sha256'],image='../../'+c['image'].replace('\\','/'),
            height=c['shape'][0],width=c['shape'][1],annotation=raw))
    (OUT/'editor.html').write_text(render_editor(manifest),encoding='utf-8')
    # Keep the user's previous editor entry point on the same simplified UI.
    old=ROOT/'experiment_outputs/standalone_geometry_v3_stage3/control_review.html'
    old.write_text(render_editor(manifest[10:]),encoding='utf-8')
    controls=json.loads((OUT/'final_control_frozen.json').read_text())['cases']
    draft=[]
    for c in controls:
        draft.append(dict(job=c['job'],sha256=c['sha256'],image='../../'+c['image'].replace('\\','/'),
            height=c['shape'][0],width=c['shape'][1],annotation=dict(sha256=c['sha256'],job_id=c['job'],source_shape=c['shape'],horizontal_polylines=[],vertical_polylines=[],human_verified=False,annotation_state='unverified_empty_source')))
    (OUT/'final_control_editor.html').write_text(render_editor(draft),encoding='utf-8')
    print('Editors built: 18 source annotations and 8 separate V3 final-control photographs')


if __name__=='__main__':main()
