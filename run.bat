@echo off
title SIH26142 - Super Resolution Mapping Console (NTRO)
echo ========================================================
echo   SIH26142: Deep Learning Super Resolution Mapping
echo   Sponsor: National Technical Research Organisation (NTRO)
echo ========================================================
echo.
echo Installing / checking requirements...
pip install -r requirements.txt
echo.
echo Launching prototype server at http://localhost:8000 ...
python run.py
pause
