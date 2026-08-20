@echo off
setlocal
set "ROOT=%~dp0"
if exist "%ROOT%release-denoise-v1.2.0-final\VideoDenoiseStudio.exe" (
  start "" "%ROOT%release-denoise-v1.2.0-final\VideoDenoiseStudio.exe"
) else if exist "%ROOT%release-denoise-v1.1.4-final\VideoDenoiseStudio.exe" (
  start "" "%ROOT%release-denoise-v1.1.4-final\VideoDenoiseStudio.exe"
) else if exist "%ROOT%release-denoise-v1.1.3-final\VideoDenoiseStudio.exe" (
  start "" "%ROOT%release-denoise-v1.1.3-final\VideoDenoiseStudio.exe"
) else if exist "%ROOT%release-denoise-v1.1.3-final\Video Denoise Studio.exe" (
  start "" "%ROOT%release-denoise-v1.1.3-final\Video Denoise Studio.exe"
) else if exist "%ROOT%release-denoise-v1.1.2-final\Video Denoise Studio.exe" (
  start "" "%ROOT%release-denoise-v1.1.2-final\Video Denoise Studio.exe"
) else if exist "%ROOT%release-denoise-v1.1.1-final\Video Denoise Studio.exe" (
  start "" "%ROOT%release-denoise-v1.1.1-final\Video Denoise Studio.exe"
) else if exist "%ROOT%release-denoise\Video Denoise Studio.exe" (
  start "" "%ROOT%release-denoise\Video Denoise Studio.exe"
) else (
  "%ROOT%work\pyinstaller311-env\Scripts\python.exe" "%ROOT%denoise_main.py"
)
