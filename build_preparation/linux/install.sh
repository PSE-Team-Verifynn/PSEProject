python3.13 -m venv venv
source venv/bin/activate
PYTHONUTF8=1
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip install git+https://github.com/Verified-Intelligence/auto_LiRPA
printf "source venv/bin/activate\nexport PYTHONPATH=$PYTHONPATH:src/\n./venv/bin/python src/nn_verification_visualisation/__main__.py" > start.sh
chmod +x start.sh
