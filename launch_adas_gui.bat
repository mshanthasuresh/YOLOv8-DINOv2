@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_adas_gui.ps1" %*
if errorlevel 1 pause