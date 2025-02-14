@echo off
REM ------------------------------------------
REM Lancement du script Python dans /webui
REM ------------------------------------------

echo Lancement de Florence-2 Batch Captioning Optimizer...
cd webui
python "Florence-2-Batch-Captioning-Optimizer.py"

REM Facultatif : garder la fenêtre ouverte en fin d'exécution
pause