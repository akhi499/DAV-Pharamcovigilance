@echo off
rem Double-click launcher: starts backend + dashboard; press Enter in the window to stop.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run.ps1" %*
if errorlevel 1 (
  echo.
  echo The launcher stopped with an error - read the message above. Logs are in the logs folder.
  pause
)
