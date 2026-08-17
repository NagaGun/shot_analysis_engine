# Runner script for FutbolConnect Shot Analysis Engine

$VENV_PYTHON = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$DRIVER_SCRIPT = Join-Path $PSScriptRoot "video_driver.py"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "Error: Virtual environment python not found at $VENV_PYTHON" -ForegroundColor Red
    Write-Host "Please ensure venv is set up properly in the repository." -ForegroundColor Yellow
    exit 1
}

Write-Host "Running FutbolConnect Shot Analysis using repo venv..." -ForegroundColor Green

if ($args.Count -ge 2) {
    # Run on a specific video clip: .\run.ps1 <video_path> <clip_id>
    & $VENV_PYTHON $DRIVER_SCRIPT $args[0] $args[1]
} elseif ($args.Count -eq 1 -and $args[0] -eq "test") {
    # Run test suite: .\run.ps1 test
    Write-Host "Running unit tests..." -ForegroundColor Cyan
    & $VENV_PYTHON -m unittest test_items_1_2.py test_v2_fixes.py
} else {
    # Run full batch on source_data videos
    Write-Host "Running batch analysis on videos in fc_juggle/source_data..." -ForegroundColor Cyan
    & $VENV_PYTHON $DRIVER_SCRIPT
}
