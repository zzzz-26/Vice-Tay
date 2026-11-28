@echo off
setlocal
rem Starts the title with scripts\play.ps1 (README.md, "Getting started").
rem Every argument goes to it, e.g.:
rem   Play-VIctory.cmd -Exe <path of your copy's executable> [-Console series-s] [-Windowed]
rem PowerShell 7 (pwsh) when it is installed, else the Windows PowerShell every Windows has.
set "PLAY_SHELL=pwsh"
where /q pwsh || set "PLAY_SHELL=powershell"
%PLAY_SHELL% -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\play.ps1" %*
exit /b %errorlevel%
