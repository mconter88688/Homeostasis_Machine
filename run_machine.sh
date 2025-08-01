#!/bin/bash

ENV_NAME="anomaly-env"
PYTHON_SCRIPT="main.py"
VENV_PATH="$HOME/$ENV_NAME/bin/activate"

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment already activated."
elif [[ -f "$VENV_PATH" ]]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$HOME/pyorbbecsdk/examples
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libatomic.so.1"
export PYTHONMALLOC=malloc

echo "Running main.py..."
python3 "$PYTHON_SCRIPT"
