# Автономные демонстрационные изображения

`demo_web_outputs/demo_flat/original.png` — искусственно созданная ровная печатная сетка.
`demo_web_outputs/demo_wave/original.png` — та же искусственная сетка после математического искажения.

Эти файлы синтетические. Они позволяют проверить, что ZIP действительно работает
без V1/V2, без старых результатов и даже без рабочего проекта. Их показатели
НЕЛЬЗЯ использовать как доказательство точности на настоящих фотографиях Торгсофта.

Пример (из корня V3):

```powershell
py -3.12 run.py benchmark --source examples/demo_web_outputs --output data/output/geometry_benchmark --run-id synthetic_demo --methods perspective,grid_mesh,tps_scipy
```

Для реальных испытаний вместо этого использовать <PROJECT_ROOT>/web_outputs.
