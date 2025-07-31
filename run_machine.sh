#!/bin/bash

ENV_NAME="anomaly-env"
PYTHON_SCRIPT="main.py"
STREAM_SCRIPT="stream_server.py"
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
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export PYTHONMALLOC=malloc
echo "Running streaming server..."
python3 "$STREAM_SCRIPT" &
STREAM_PID=$!


echo "Running main logic script in a new terminal..."
gnome-terminal -- bash -c "python3 \"$PYTHON_SCRIPT\"; echo 'Main script ended'; read -p 'Press Enter to close...'" &


trap "echo 'Terminating...'; kill $STREAM_PID; exit" SIGINT


sleep 2
if command -v xdg-open > /dev/null; then
    xdg-open http://127.0.0.1:5000/video
elif command -v open > /dev/null; then
    open http://127.0.0.1:5000/video
else
    echo "Visit http://127.0.0.1:5000/video"
fi


wait $STREAM_PID
