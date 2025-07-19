#!/bin/bash

ENV_NAME = "anomaly-env"
PYTHON_SCRIPT = "main.py"

if [[{ "$VIRTUAL_ENV" != ""}]]; then
    echo "Virtual Env Already On"
else
    echo "Activating Virtual Env"
    source ~/$ENV_NAME/bin/activate
fi

echo "Running Python Script"
python3 $PYTHON_SCRIPT

echo "Deactivating Virtual Env"
deactivate

