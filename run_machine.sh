#!/bin/bash

ENV_NAME="anomaly-env"
PYTHON_SCRIPT="main.py"

# Check if virtual environment is already activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment already activated."
else
    echo "Activating virtual environment..."
    source ~/"$ENV_NAME"/bin/activate
fi

#export QT_QPA_PLATFORM=eglfs

echo "Running Python script..."
python3 "$PYTHON_SCRIPT"

