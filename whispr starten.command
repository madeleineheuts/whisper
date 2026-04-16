#!/bin/bash
# whispr starten – Terminal schließt sich danach automatisch

cd "$(dirname "$0")"

# Env-Variablen laden falls .env existiert
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# whispr im Hintergrund starten, Logs in whispr.log
nohup python3 whispr.py >> whispr.log 2>&1 &

# Terminal-Fenster schließen
osascript -e 'tell application "Terminal" to close front window' &
