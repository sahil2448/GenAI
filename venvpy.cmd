@echo off
setlocal

rem Force temp dirs into the repo so sandboxed runs (and ensurepip) can write.
rem Avoid dot-prefixed dirs here; some Windows setups end up with odd ACLs.
set "TEMP=%~dp0tmp"
set "TMP=%~dp0tmp"
set "TMPDIR=%~dp0tmp"

if not exist "%~dp0tmp" mkdir "%~dp0tmp" >nul 2>nul

"%~dp0.venv\Scripts\python.exe" %*
