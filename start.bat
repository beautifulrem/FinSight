@echo off
setlocal
cd /d "%~dp0"

set "BUNDLED_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%BUNDLED_PYTHON%" (
  set "PYTHON_CMD="%BUNDLED_PYTHON%""
) else (
  where python >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=python"
    echo Bundled .venv was not found. Falling back to system Python.
  ) else (
    where py >nul 2>nul
    if not errorlevel 1 (
      set "PYTHON_CMD=py -3"
      echo Bundled .venv was not found. Falling back to py launcher.
    ) else (
      echo Missing bundled Python environment: %BUNDLED_PYTHON%
      echo Python was not found on PATH either.
      echo Please run scripts\build_windows_dist.ps1 first, or install Python and run pip install -r requirements.txt.
      pause
      exit /b 1
    )
  )
)

%PYTHON_CMD% "%~dp0scripts\launch_chatbot.py"
if errorlevel 1 (
  echo Financial Chatbot stopped with an error.
  pause
  exit /b 1
)
