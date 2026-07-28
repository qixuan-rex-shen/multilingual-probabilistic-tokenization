@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."

"%PROJECT_ROOT%\.venv\Scripts\pythonw.exe" "%PROJECT_ROOT%\scripts\run_remaining_pipeline_resume.py" --bpe-worker-pid %1 --poll-seconds 300
