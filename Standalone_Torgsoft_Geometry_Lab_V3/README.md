# Standalone Torgsoft Geometry Lab V3

**Полный самостоятельный Trial. Не требуется V1, V2, `trial_geometry`, Stage A/B/C/D,
старые JSON, старые отчёты, старые модели или загрузка чужих GitHub-репозиториев.**

Задачи:
1. Приводить печатную товарную таблицу к прямоугольному виду.
2. По **той же** геометрической карте размещать синюю сетку строго на границах ячеек.
3. Сопоставлять результаты только после независимой проверки на реальных фото и GT.

В ZIP включены **реальные исходники**, а не только ТЗ:
- `run.py` — работающая CLI, обнаружение job, дедупликация, запуск, отчёты;
- `geometry_lab/cases.py` — чтение текущего `web_outputs`, baseline при наличии;
- `geometry_lab/detection.py` — собственный независимый поиск печатных линий OpenCV;
- `geometry_lab/algorithms.py` — три рабочих геометрических метода;
- `geometry_lab/metrics.py` — проверка якобиана, совпадения с GT;
- `geometry_lab/reports.py` — HTML/JSON отчёт и ссылки на изображения;
- `geometry_lab/synthetic.py` — автономные тестовые данные;
- `annotate.html` — офлайн-редактор ручной разметки;
- `tests/` — автоматические тесты.

## Установка Windows PowerShell

Распаковать единственную корневую папку архива в корень проекта `Product_Matcher`.
Папка может находиться и вне репозитория, тогда передайте `--project-root` или `--source`.

```powershell
cd .\Standalone_Torgsoft_Geometry_Lab_V3
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-optional-tps.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe run.py discover --max-cases 10
.\.venv\Scripts\python.exe run.py benchmark --max-cases 10
```

`pytest` не входит в runtime-файл зависимостей, поэтому отдельно для тестов:
```powershell
.\.venv\Scripts\python.exe -m pip install pytest
```

Минимальная установка без SciPy/TPS:
```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 run.py benchmark --methods baseline,perspective,grid_mesh --max-cases 10
```

По умолчанию вход: **`../web_outputs/<job_id>/`**;
выход: **`../data/output/geometry_benchmark/standalone_YYYYMMDD_HHMMSS/`**.

Пример работы из другого каталога:
```powershell
py -3.12 run.py benchmark --project-root "C:\Work\Prj_35_OCR_prep" --max-cases 0 --run-id full_first
```

## Важно

- Источник `web_outputs` **только для чтения**. Запрещены запись, очистка и миграция.
- Никакие рабочие модули `App/` не импортируются и не редактируются.
- Исходное фото — ориентированное `original.png` (если есть), иначе байты `upload.bin`.
- Baseline берётся непосредственно из текущего основного OCR: `rectified.png`,
  `rectification_map.npz`, `layout.json`. При отсутствии => `NOT_AVAILABLE`, не ошибка.
- Исходники V1/V2, старые результаты и исторические Trial **не используются**.
- Сравнение: `baseline`, `perspective`, `grid_mesh`, `tps_scipy`.
- Повторы исходных фото исключаются по SHA256 байтов `upload.bin`.
- Удачный тест, безопасная карта и красивые линии **не доказывают** корректного попадания
  синей разметки в реальные печатные границы.
- Для измерения качества требуется новая независимая ручная разметка:
  открывайте `annotate.html`, выбирайте `original.png`, сохраняйте `ground_truth/<sha256>.json`.
- Отчёт `summary.html` автоматически отличает `GT_MISSING`, `GT_MISMATCH`,
  `REJECTED`, `NOT_AVAILABLE`, `SUCCESS`, `GEOMETRY_WARNING`.
- На перспективных фото и сильных складках метод может отклонить результат — это безопаснее,
  чем ложно заявить, что найдено пять верных границ.

**Для ИИ-программиста первоочередной файл: `AGENT_TASK.md`.**

## Автономная проверка без доступа к рабочему проекту

В `examples/demo_web_outputs` есть две **синтетические** фотографии, включённые в ZIP:
```powershell
.\.venv\Scripts\python.exe run.py benchmark --source examples/demo_web_outputs --output data/output/geometry_benchmark --run-id synthetic_demo --methods perspective,grid_mesh,tps_scipy
```

Этот пример подтверждает только работоспособность пакета и делает ZIP полностью
самодостаточным для технического smoke-теста. Для проверки применимости к реальным
чекам переключите вход на настоящий `web_outputs`.
