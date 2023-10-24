#!/bin/sh

if [ ! -d venv ]; then
    virtualenv --python=python3 venv
fi

. .venv/bin/activate
pip install -r requirements.txt

cp torch_optim_files/compressedSGD.py compressedSGD.pyi __init__.py __init__.pyi venv/lib/python3.11/site-packages/torch/optim
