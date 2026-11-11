@echo off
chcp 65001 >nul
setlocal EnableExtensions
title Prj_17 Matching Product - server

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] Не найдено окружение .venv\Scripts\python.exe
    echo Установите зависимости по README.md и запустите файл снова.
    pause
    exit /b 1
)

rem Порт берём из .env, иначе 8017.
set "PORT=8017"
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /R /C:"^OCR_WEB_PORT=" ".env" 2^>nul`) do set "PORT=%%B"
set "PORT=%PORT: =%"
set "PORT=%PORT:"=%"
if not defined PORT set "PORT=8017"

echo [1/3] Останавливаю сервер на порту %PORT% ...
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"`) do (
    echo       завершаю процесс PID %%P
    taskkill /F /T /PID %%P >nul 2>&1
)

rem На всякий случай завершаем оставшиеся экземпляры этого сервера.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'App\.web_server' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo [2/3] Ожидаю освобождения порта %PORT% ...
powershell -NoProfile -Command "$end = (Get-Date).AddSeconds(15); while ((Get-Date) -lt $end -and (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue)) { Start-Sleep -Milliseconds 300 }; if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo [ОШИБКА] Порт %PORT% всё ещё занят. Закройте процесс вручную и повторите.
    pause
    exit /b 1
)

echo [3/3] Запускаю сервер: http://127.0.0.1:%PORT%
echo Для остановки закройте это окно или нажмите Ctrl+C.
echo.
".venv\Scripts\python.exe" -m App.web_server

set "EXITCODE=%ERRORLEVEL%"
echo.
echo Сервер завершился с кодом %EXITCODE%.
pause
exit /b %EXITCODE%
