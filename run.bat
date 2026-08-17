@echo off
set VENV_PYTHON=%~dp0venv\Scripts\python.exe
set DRIVER_SCRIPT=%~dp0video_driver.py

if not exist "%VENV_PYTHON%" (
    echo Error: Virtual environment python not found at %VENV_PYTHON%
    exit /b 1
)

echo Running FutbolConnect Shot Analysis using repo venv...

if "%~1"=="test" (
    "%VENV_PYTHON%" -m unittest test_items_1_2.py test_v2_fixes.py
) else if not "%~2"=="" (
    "%VENV_PYTHON%" "%DRIVER_SCRIPT%" "%~1" "%~2"
) else (
    "%VENV_PYTHON%" "%DRIVER_SCRIPT%"
)
