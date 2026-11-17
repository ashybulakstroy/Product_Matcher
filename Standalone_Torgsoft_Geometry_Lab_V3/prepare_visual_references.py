from pathlib import Path
import json
R=Path.cwd();lab=R/'Standalone_Torgsoft_Geometry_Lab_V3';selected=json.loads((R/'experiment_outputs/standalone_geometry_v3/selected.json').read_text());byjob={c['job']:c for c in selected}
# Independent visual annotations on original photographs; no model/grid output used.
cases={
'2efa41ac6b3d45ca9457e456b5768cd5':('mild_perspective',[
[[66,672],[102,670],[370,661],[417,658],[504,652]],
[[70,741],[106,738],[376,729],[421,726],[507,721]],
[[74,789],[110,785],[379,778],[425,775],[509,770]],
[[78,838],[113,834],[382,827],[427,824],[511,819]],
[[80,866],[115,863],[384,855],[429,852],[514,848]],
[[83,916],[118,912],[389,904],[435,901],[517,896]]]),
'112d84af7d11404d9005d4c64e4591d3':('torn_occlusion',[
[[82,434],[168,439],[537,459],[651,465],[813,475]],
[[81,517],[165,522],[533,541],[646,548],[806,556]],
[[82,634],[164,641],[527,660],[640,667],[798,674]],
[[84,696],[164,704],[523,722],[636,728],[794,734]]]),
'b23a7ad5e57a41a098567945b4270e65':('near_flat',[
[[96,520],[156,520],[475,519],[582,518],[737,518]],
[[95,594],[155,594],[474,594],[580,594],[735,595]],
[[95,663],[154,664],[473,667],[579,668],[733,670]],
[[94,718],[154,720],[472,723],[577,725],[731,727]]]),
'ac2fd8bce9754c649af831b3b7139cf2':('mild_curved',[]),
'4eae80e83af6442b817f03422d60c0e4':('wrinkled_partial_visibility',[
[[75,296],[125,302],[365,317],[440,312],[552,300]],
[[73,349],[123,355],[364,367],[440,362],[551,352]],
[[71,389],[121,396],[364,405],[440,401],[551,394]],
[[69,430],[118,438],[363,444],[440,440],[552,434]],
[[65,508],[114,514],[362,519],[440,516],[552,512]],
[[61,567],[110,570],[360,574],[440,573],[553,570]],
[[55,672],[103,676],[358,679],[439,679],[556,677]],
[[49,727],[98,731],[357,735],[439,733],[558,730]],
[[43,781],[94,787],[355,792],[440,789],[559,786]],
[[34,836],[86,842],[353,848],[440,847],[562,844]],
[[25,895],[77,902],[351,909],[440,909],[565,906]],
[[10,989],[64,996],[350,1004],[441,1007],[568,1006]],
[[0,1055],[56,1063],[349,1074],[442,1073],[571,1071]],
[[0,1110],[53,1113],[348,1123],[442,1125],[574,1124]]])}
ys=[[339,336,330,326,322],[381,379,371,368,365],[416,414,403,401,397],[450,446,436,433,430],[507,504,493,490,486],[550,545,535,532,528],[634,631,621,618,613],[677,674,665,662,657],[720,716,707,704,699],[763,761,752,750,747],[824,820,814,811,807],[889,886,880,878,875],[923,921,915,912,908]]
for row in ys:
 t=(sum(row)/5-331)/(916-331)
 xs=[round(a+t*(b-a)) for a,b in zip([113,150,343,403,490],[139,177,369,430,520])]
 cases['ac2fd8bce9754c649af831b3b7139cf2'][1].append([[x,y] for x,y in zip(xs,row)])
for job,(kind,nodes) in cases.items():
 c=byjob[job];vertical=[[row[i] for row in nodes] for i in range(5)]
 if kind=='wrinkled_partial_visibility':vertical[0]=vertical[0][:-1]
 data=dict(sha256=c['sha256'],job_id=job,source_shape=c['shape'],vertical_polylines=vertical,horizontal_polylines=nodes,
 annotation_origin='agent_visual_source_only',human_verified=False,estimated_uncertainty_px=4,
 stratum=kind,limitations='Preliminary independent visual tracing; not human-validated GT. No ground-truth coordinates taken from baseline or detector.',
 missing_strata=['strong_low_angle','sharp_fold'])
 (lab/'ground_truth'/f"{c['sha256']}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('5 preliminary visual reference sets; no claim of human-verified GT')
