"""Single editor contract for subsequent Geometry Lab visual tools."""
import json
from pathlib import Path


def render_editor(manifest):
    normalized=[]
    for source in manifest:
        m=dict(source)
        if 'annotation' not in m:
            if 'reference'in m:m['annotation']=m['reference']
            else:
                nodes=m.get('nodes',[])
                m['annotation']=dict(sha256=m['sha256'],job_id=m['job'],nodes=nodes,
                    source_shape=[m.get('height'),m.get('width')],
                    horizontal_polylines=nodes,vertical_polylines=[[r[c] for r in nodes] for c in range(5)] if nodes else [],
                    human_verified=False,annotation_state='automatic_proposals_unverified')
        normalized.append(m)
    return (Path(__file__).parent/'grid_editor.html').read_text(encoding='utf-8').replace('__MANIFEST__',json.dumps(normalized,ensure_ascii=False))
