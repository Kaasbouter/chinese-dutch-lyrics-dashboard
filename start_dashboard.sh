#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.10 or newer is required."
    echo "Install Python from https://www.python.org/downloads/ and try again."
    exit 1
fi

python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "Python 3.10 or newer is required."
    exit 1
}

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating the private local Python environment..."
    python3 -m venv .venv
fi

if [ ! -f ".venv/.lyrics-dashboard-ready" ]; then
    echo "Installing or checking the free local dependencies..."
    ".venv/bin/python" -m pip install --upgrade pip
    ".venv/bin/python" -m pip install -r requirements.txt
    ".venv/bin/python" -m pip check
    printf '%s\n' "setup complete" > ".venv/.lyrics-dashboard-ready"
fi

echo
echo "Starting the Chinese-Dutch Lyrics Converter..."
echo "Keep this terminal open while using the dashboard."
echo "Press Ctrl+C to stop it."
echo
exec ".venv/bin/python" -m streamlit run app.py
