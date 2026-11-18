param(
    [ValidateSet('fast', 'ocr', 'full', 'targeted')]
    [string]$Mode = 'fast',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Targets
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host '[ERROR] Project .venv not found. See README.md.' -ForegroundColor Red
    exit 1
}

if ($Mode -eq 'fast' -or $Mode -eq 'ocr') {
    & $python -c "import xdist" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[ERROR] pytest-xdist is not installed.' -ForegroundColor Red
        Write-Host 'Run: .\.venv\Scripts\python.exe -m pip install -r requirements-test.txt'
        exit 1
    }
}

$pytestArgs = @('-m', 'pytest', '-q')
switch ($Mode) {
    'fast' { $pytestArgs += @('App/tests', '-n', '5', '--dist', 'worksteal') }
    'ocr'  { $pytestArgs += @('App/tests/ocr', '-n', '5', '--dist', 'worksteal') }
    'full' { $pytestArgs += @('App/tests') }
    'targeted' {
        if (-not $Targets -or $Targets.Count -eq 0) {
            Write-Host '[ERROR] Mode targeted requires at least one test path.' -ForegroundColor Red
            Write-Host 'Example: .\run_tests.ps1 -Mode targeted App/tests/test_matching.py'
            exit 2
        }
        $pytestArgs += $Targets
    }
}

Write-Host ("[tests] mode={0}" -f $Mode) -ForegroundColor Cyan
Write-Host ("[tests] {0} {1}" -f $python, ($pytestArgs -join ' ')) -ForegroundColor DarkGray

$sw = [Diagnostics.Stopwatch]::StartNew()
& $python @pytestArgs
$code = $LASTEXITCODE
$sw.Stop()

if ($code -eq 0) {
    Write-Host ("[tests] elapsed={0:N1}s exit={1}" -f $sw.Elapsed.TotalSeconds, $code) -ForegroundColor Green
} else {
    Write-Host ("[tests] elapsed={0:N1}s exit={1}" -f $sw.Elapsed.TotalSeconds, $code) -ForegroundColor Red
}
exit $code
