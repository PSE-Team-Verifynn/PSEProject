python3.13 -m venv venv
source venv/bin/activate
./venv/bin/python -m pip install -r requirements.txt
printf "source venv/bin/activate\nexport PYTHONPATH=$PYTHONPATH:src/\n./venv/bin/python src/nn_verification_visualisation/__main__.py" > start.sh
chmod +x start.sh
