#!/bin/bash
echo "📦 Installiere Basis-Abhängigkeiten..."
pip3 install faster-whisper sounddevice pynput numpy flask rumps anthropic soundfile Pillow requests

echo ""
echo "📦 Installiere pyannote (Speaker Diarization)..."
pip3 install pyannote.audio

echo ""
echo "✅ Fertig! Vor dem Start API Keys setzen:"
echo ""
echo "   export ANTHROPIC_API_KEY='sk-ant-...'"
echo "   export HF_TOKEN='hf_...'"
echo "   export NOTION_TOKEN='secret_...'     # optional, nur für Test-Export"
echo ""
echo "Dann starten:"
echo "   python3 whispr.py"
echo ""
echo "Oder einmalig direkt:"
echo "   ANTHROPIC_API_KEY='sk-ant-...' HF_TOKEN='hf_...' python3 whispr.py"
