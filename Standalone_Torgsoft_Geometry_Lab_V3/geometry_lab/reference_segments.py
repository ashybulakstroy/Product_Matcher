"""Trim reference tails by user-defined graph connectivity, without moving nodes."""
import copy


def intersection_bounded_reference(source):
    result=copy.deepcopy(source)
    sets={a:set(n for line in source[a+'_node_ids'] for n in line) for a in ['horizontal','vertical']}
    edits=[];incomplete=[]
    for axis,other in [('horizontal','vertical'),('vertical','horizontal')]:
        lines=[]
        for i,line in enumerate(source[axis+'_node_ids']):
            shared=[j for j,n in enumerate(line) if n in sets[other]]
            if len(shared)<2:
                incomplete.append(dict(axis=axis,line=i,reason='fewer_than_two_perpendicular_intersections'))
                continue
            first,last=min(shared),max(shared)
            lines.append(line[first:last+1])
            edits.append(dict(axis=axis,line=i,removed_before=line[:first],removed_after=line[last+1:],
                              supported_intersections=len(shared)))
        # Sort line families only; retain source order and exact coordinates along
        # each physical polyline. No snapping, interpolation or missing junctions.
        transverse='y' if axis=='horizontal' else 'x'
        lines.sort(key=lambda ids:sum(source['graph_nodes'][n][transverse] for n in ids)/len(ids))
        result[axis+'_node_ids']=lines
        result[axis+'_polylines']=[[[source['graph_nodes'][n]['x'],source['graph_nodes'][n]['y']] for n in ids] for ids in lines]
    used=set(n for a in ['horizontal','vertical'] for line in result[a+'_node_ids'] for n in line)
    result['graph_nodes']={n:copy.deepcopy(source['graph_nodes'][n]) for n in used}
    result.pop('nodes',None)  # The original matrix contains the editor's tails.
    result['reference_scope']='segments_bounded_by_existing_perpendicular_links'
    result['confirmation_provenance']='User explicitly confirmed perpendicular-linked segments; free tails excluded.'
    result['source_human_verified']=source.get('human_verified',False)
    result['human_verified']=False  # Do not relabel the original export globally.
    result['user_confirmed_scope']=True
    result['tail_normalization']=dict(edits=edits,excluded=incomplete,coordinates_moved=0,
        removed_node_occurrences=sum(len(e['removed_before'])+len(e['removed_after']) for e in edits))
    hsets=[set(x) for x in result['horizontal_node_ids']];vsets=[set(x) for x in result['vertical_node_ids']]
    result['complete_intersection_matrix']=bool(len(vsets)==5 and hsets and all(len(h&v)==1 for h in hsets for v in vsets))
    return result
