#!/bin/sh

if [ ! -d venv ]; then
    virtualenv --python=python3 venv
fi

. .venv/bin/activate
pip install -r requirements.txt

cp torch_optim_files/* venv/lib/python3.11/site-packages/torch/optim
