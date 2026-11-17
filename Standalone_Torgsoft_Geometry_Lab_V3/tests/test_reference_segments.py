from geometry_lab.reference_segments import intersection_bounded_reference


def test_trim_only_free_tails_and_preserve_exact_coordinates():
    source=dict(graph_nodes={n:dict(x=x,y=y) for n,x,y in [('a',0,10),('b',20,10),('c',50,11),('d',80,10),('e',100,10),('u',20,0),('v',20,30),('w',80,0),('z',80,30)]},
        horizontal_node_ids=[['a','b','c','d','e']],vertical_node_ids=[['u','b','v'],['w','d','z']],human_verified=False)
    # Two intersections per vertical are needed, so add a second real row.
    source['graph_nodes']['b2']=dict(x=20,y=20);source['graph_nodes']['d2']=dict(x=80,y=20)
    source['horizontal_node_ids'].append(['b2','d2']);source['vertical_node_ids']=[['u','b','b2','v'],['w','d','d2','z']]
    out=intersection_bounded_reference(source)
    assert out['horizontal_node_ids'][0]==['b','c','d']
    assert out['vertical_node_ids'][0]==['b','b2']
    assert out['graph_nodes']['c']==source['graph_nodes']['c']
    assert source['horizontal_node_ids'][0][0]=='a' and 'a'in source['graph_nodes']
    assert out['human_verified'] is False and out['user_confirmed_scope'] is True


def test_no_intersection_is_not_invented():
    source=dict(graph_nodes={'a':dict(x=0,y=0),'b':dict(x=10,y=0),'c':dict(x=5,y=-10),'d':dict(x=5,y=10)},horizontal_node_ids=[['a','b']],vertical_node_ids=[['c','d']])
    out=intersection_bounded_reference(source)
    assert out['horizontal_node_ids']==[] and out['vertical_node_ids']==[]
    assert out['tail_normalization']['coordinates_moved']==0
