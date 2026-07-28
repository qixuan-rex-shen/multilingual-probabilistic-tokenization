@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."

"%PROJECT_ROOT%\.venv\Scripts\pythonw.exe" "%PROJECT_ROOT%\scripts\resume_bpe_mlm.py"
