@echo off
REM ------------------------------------------
REM Launching Florence-2 script inside /webui
REM ------------------------------------------

cd /d "%~dp0webui"
echo Installing dependencies...
pip install -r ../requirements.txt

echo Starting Florence-2 Batch Captioning Optimizer...
python "F2BC-Optimizer.py"

REM Optional: Keep the window open to view logs
pause
