@echo off
py -3.13 -m venv venv
call venv\Scripts\activate.bat
set PYTHONUTF8=1
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -c "import torch; print(torch.__version__)"
venv\Scripts\python -m pip install git+https://github.com/Verified-Intelligence/auto_LiRPA --no-build-isolation
(
echo @echo off
echo call venv\Scripts\activate.bat
echo set PYTHONPATH=%%PYTHONPATH%%;src
echo venv\Scripts\python src\nn_verification_visualisation\__main__.py
) > start.bat
pause