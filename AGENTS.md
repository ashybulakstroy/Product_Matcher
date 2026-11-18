# AGENTS.md

Инструкции для AI-агентов (Codex, Claude и другие), работающих с репозиторием
`Prj_17_Matching_Product`.

## Тестирование

Все команды выполнять из корня репозитория и только проектным интерпретатором
`.\.venv\Scripts\python.exe`.

### 1. Быстрый прогон — режим по умолчанию

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_tests.ps1 -Mode fast
```

Эквивалент:

```powershell
.\.venv\Scripts\python.exe -m pytest App/tests -q -n 5 --dist worksteal
```

- Ожидаемый результат: **549 passed, 16 skipped, 0 failed**.
- Время на текущей машине (i5-13400, 10 ядер): **~10–11 секунд**.
- Это основной режим для проверки правок и перед финальным ответом.

### 2. Smoke-прогон на ~5 секунд (только OCR)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_tests.ps1 -Mode ocr
```

Эквивалент:

```powershell
.\.venv\Scripts\python.exe -m pytest App/tests/ocr -q -n 5 --dist worksteal
```

- Ожидаемый результат: **356 passed, 15 skipped, 0 failed**.
- Время: **~4,5–5,5 секунды** (зависит от загрузки машины).
- Использовать для быстрой проверки OCR-кода. Это **не** замена полному набору:
  тесты ядра приложения (`App/tests/*.py`) в него не входят.

### 3. Эталонный полный прогон (последовательно)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_tests.ps1 -Mode full
```

Эквивалент:

```powershell
.\.venv\Scripts\python.exe -m pytest App/tests -q
```

- Ожидаемый результат: **549 passed, 16 skipped, 0 failed**.
- Время: **~15–16 секунд** (после оптимизации teardown).
- Использовать перед коммитом, при подозрении на гонки и когда нужен
  воспроизводимый эталонный результат без параллельности.

### 4. Точечная проверка изменённых тестов

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_tests.ps1 -Mode targeted App/tests/test_matching.py
```

Можно передавать несколько путей или node id:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_tests.ps1 -Mode targeted App/tests/test_matching.py::test_match_products_returns_top_candidates_sorted_by_similarity
```

- Время: обычно **0,1–2 секунды** на модуль.
- При правке одного модуля сначала запускать именно его, полный набор — после.

## Обязательные правила

1. **По умолчанию использовать `-Mode fast`** (`-n 5 --dist worksteal`).
   Не запускать `-n auto`, `-n 8` или `-n 10`: на этой машине они медленнее
   (~10 с и больше) из-за конкуренции процессов.
2. **Не использовать `-n 4`.** При 4 воркерах наблюдался флейк
   `test_manual_mappings::test_parallel_writers_and_manual_text_file_changes`
   (`PermissionError` на `knowledge.json`). Рабочий минимум — `-n 5`.
3. **Не запускать два прогона pytest одновременно.** Тесты пишут во временные
   каталоги и общий `knowledge.json`; параллельные запуски могут конфликтовать.
4. **При падении в параллельном режиме сначала перепроверить упавший тест
   последовательно**, и только потом считать это дефектом продукта:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest "App/tests/...::test_name" -q
   ```

5. **Не откатывать `poll_interval=0.05`** в тестовых фикстурах. Это осознанная
   оптимизация: teardown HTTP-серверов ускорен с 0,5 с до 0,05 с
   (файлы: `test_web_server.py`, `test_matching_bridge_integration.py`,
   `test_full_results.py`, `test_cancel_job.py`, `test_cancel_job_extended.py`).
6. **Сервер на порту 8017 останавливать не нужно.** Тесты поднимают свои
   серверы на свободном порту (`port 0`).
7. **`pytest-xdist` обязателен для быстрых режимов.** Если его нет:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
   ```

## Ожидаемые результаты и отчёт

После прогона агент обязан сообщить:

- режим запуска;
- количество `passed` / `skipped` / `failed`;
- время выполнения;
- при падениях — имя теста и первую значимую ошибку.

Норма для этого репозитория: **549 passed, 16 skipped, 0 failed**.
`16 skipped` — ожидаемые пропуски, а не проблема.

## Что не делать

- Не «ускорять» тесты удалением или пропуском проверок.
- Не менять рабочий код ради зелёного теста без объяснения причины.
- Не коммитить изменения, если полный прогон (`-Mode full`) не зелёный.
- Не запускать тесты через системный Python — только `.\.venv\Scripts\python.exe`.
