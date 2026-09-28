@echo off
rem Convenience wrapper: run.cmd <args...> forwards to asmr_spatial.cli
setlocal
set "PYTHONPATH=%~dp0src"
"%~dp0.venv\Scripts\python.exe" -m asmr_spatial.cli %*
