"""Canonical intersection-matrix annotations; each node belongs to two lines."""
import numpy as np


def export_lines(nodes):
    a=np.asarray(nodes,dtype=float)
    if a.ndim!=3 or a.shape[1:]!=(5,2) or len(a)<3 or not np.isfinite(a).all():
        raise ValueError('expected at least 3 rows of 5 finite intersection points')
    if np.min(np.diff(a[:,:,0],axis=1))<=1 or np.min(np.diff(a[:,:,1],axis=0))<=1:
        raise ValueError('crossing or unordered nodes')
    return dict(nodes=a.tolist(),horizontal_polylines=a.tolist(),
                vertical_polylines=[a[:,c].tolist() for c in range(5)])


def validate_linked_annotation(annotation):
    if annotation.get('schema')=='geometry_grid_graph_v1':
        points=annotation['graph_nodes']
        for axis in ['horizontal','vertical']:
            lines=[[[points[n]['x'],points[n]['y']] for n in ids] for ids in annotation[axis+'_node_ids']]
            if lines!=annotation.get(axis+'_polylines'):raise ValueError('graph and exported polylines disagree')
            for line in lines:
                p=np.asarray(line,float)
                along=0 if axis=='horizontal' else 1
                if len(p)<2 or not np.isfinite(p).all() or np.any(np.diff(p[:,along])<=0):
                    raise ValueError('non-monotone or invalid graph line')
        return {k:annotation.get(k) for k in ['nodes','horizontal_polylines','vertical_polylines']}
    expected=export_lines(annotation['nodes'])
    for key in ['horizontal_polylines','vertical_polylines']:
        if annotation.get(key)!=expected[key]:raise ValueError('lines disagree with canonical shared nodes')
    return expected
