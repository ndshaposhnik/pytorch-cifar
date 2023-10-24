#!/bin/sh

if [ ! -d venv ]; then
    virtualenv --python=python3 venv
fi

. .venv/bin/activate
pip install -r requirements.txt

cp compressedSGD.py venv/lib/python3.11/site-packages/torch/optim
cp compressedSGD.pyi venv/lib/python3.11/site-packages/torch/optim
cp __init__.py venv/lib/python3.11/site-packages/torch/optim
cp __init__.pyi venv/lib/python3.11/site-packages/torch/optim
