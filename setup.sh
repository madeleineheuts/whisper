#!/bin/bash
echo "📦 Installiere Abhängigkeiten..."
pip3 install faster-whisper sounddevice pynput numpy flask rumps
echo ""
echo "✅ Fertig! Starte whispr mit:"
echo "   python3 whispr.py"
