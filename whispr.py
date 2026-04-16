#!/usr/bin/env python3
"""
whispr.py — Voice Dictation Menu Bar App für Mac
─────────────────────────────────────────────────
Modus Halten  (Standard): fn gedrückt halten → aufnehmen → loslassen → einfügen
Modus Toggle  (im Menü):  fn einmal drücken  → aufnehmen → fn nochmal → einfügen

Sprache (im Menü wechselbar): DE | EN

Setup: pip3 install faster-whisper sounddevice pynput numpy flask rumps
Run:   python3 whispr.py

Voraussetzung: Accessibility-Zugriff für Terminal/Python in
Systemeinstellungen → Datenschutz & Sicherheit → Bedienungshilfen
"""

import rumps
import sounddevice as sd
import numpy as np
import subprocess
import threading
import sqlite3
import json
import csv
import io
import sys
import os
import time
import tempfile
from datetime import date, datetime
from flask import Flask, Response, jsonify, request
from faster_whisper import WhisperModel
from pynput import keyboard

try:
    import anthropic as _anthropic_lib
    _ANTHROPIC = True
except ImportError:
    _ANTHROPIC = False

try:
    import soundfile as _sf
    _SOUNDFILE = True
except ImportError:
    _SOUNDFILE = False

try:
    from PIL import Image as _PILImage
    _PIL = True
except ImportError:
    _PIL = False

try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    from pyannote.audio import Pipeline as _PyannotePipeline
    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False

# AppKit (via pyobjc — bereits durch rumps installiert)
try:
    from AppKit import (
        NSPanel, NSColor, NSMakeRect, NSTextField, NSFont,
        NSView, NSBezierPath, NSNonactivatingPanelMask,
        NSBorderlessWindowMask, NSFloatingWindowLevel,
        NSBackingStoreBuffered, NSScreen,
    )
    from Foundation import NSObject
    import objc as _objc
    _APPKIT = True
except Exception:
    _APPKIT = False

# ─── Config ───────────────────────────────────────────────
SAMPLE_RATE = 16000
MODEL_SIZE  = "small"   # tiny | small | medium
LANGUAGE    = "de"      # "de" | "en"
VAD_FILTER  = True      # Voice Activity Detection (Stille rausfiltern)
PORT        = 5173
DB_PATH     = os.path.expanduser("~/.whispr.db")

# API Keys — als Umgebungsvariablen setzen oder hier direkt eintragen
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HF_TOKEN          = os.environ.get("HF_TOKEN", "")        # HuggingFace für pyannote
NOTION_TOKEN      = os.environ.get("NOTION_TOKEN", "")    # Notion Internal Integration Token

# Notion: Parent-Page für alle Test-Reports (Testing & Feedback unter Product)
NOTION_TESTING_PAGE_ID = "33ea32b52b49817ab561fc6d88dd991b"
NOTION_BUG_DB_ID       = "7aaad96c65ac48b6a949bf99adea5e72"  # Bug & Feedback Tracker v2
MADELEINE_NOTION_ID    = "0cd9e3be-e325-495b-90ee-e7bb73694782"

# Bekannte Call-Apps → Typ (external / internal)
CALL_APPS = {
    "zoom.us":           "external",
    "Zoom":              "external",
    "Microsoft Teams":   "external",
    "Webex":             "external",
    "FaceTime":          "external",
    "Slack":             "internal",
}
# Browser-URLs die als Call gelten
CALL_URLS = ["meet.google.com", "app.gather.town", "whereby.com", "teams.microsoft.com/meet"]
# ──────────────────────────────────────────────────────────

# Hotkey für Aufnahme — rechte Option-Taste (fn funktioniert nicht auf neueren Macs)
FN_KEY = keyboard.Key.alt_r

# ─── Database ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dictations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL,
                word_count INTEGER DEFAULT 0,
                wpm        INTEGER DEFAULT 0,
                duration   REAL    DEFAULT 0,
                created_at TEXT    DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS dictionary (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                word       TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS snippets (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger    TEXT NOT NULL UNIQUE,
                expansion  TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meetings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                app        TEXT,
                call_type  TEXT,
                transcript TEXT,
                summary    TEXT,
                duration   REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
        """)
        conn.commit()

def get_setting(key: str, default: str = "") -> str:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default

def set_setting(key: str, value: str):
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)", (key, value))
        conn.commit()

init_db()

# Persistierte Einstellungen laden
def _load_settings():
    global LANGUAGE, VAD_FILTER
    lang = get_setting("language", LANGUAGE)
    if lang in ("de", "en"):  # "auto" nicht mehr unterstützt → default de
        LANGUAGE = lang
    elif lang == "auto":
        LANGUAGE = "de"
        set_setting("language", "de")
    vad = get_setting("vad_filter", "1")
    VAD_FILTER = vad == "1"

_load_settings()

# ─── Whisper ──────────────────────────────────────────────
print("⏳ Lade Whisper-Modell...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"✅ Modell geladen ({MODEL_SIZE})")

# ─── Audio state ──────────────────────────────────────────
recording    = False
fn_pressed   = False
toggle_mode  = False   # False = Halten, True = einmal drücken
audio_frames = []
record_start = None
stream_lock  = threading.Lock()
stream       = None    # wird nur beim Aufnehmen geöffnet
last_text    = None    # letztes Transkript (für "Nochmal einfügen")
_last_level_t = 0.0    # Throttle für Level-Updates im Overlay

# Overlay-Instanz (wird nach Klassen-Definition weiter unten erzeugt)
overlay: "RecordingOverlay | None" = None

# ─── Meeting State ────────────────────────────────────────
meeting_active  = False
meeting_frames  = []
meeting_app     = ""
meeting_type    = ""
meeting_start   = None
meeting_stream  = None
meeting_lock    = threading.Lock()

# ─── Call Detection ───────────────────────────────────────
def get_active_call_app():
    """Prüft ob ein aktiver Call läuft. Gibt (app_name, call_type) zurück."""
    try:
        r = subprocess.run(
            ["osascript", "-e",
             "tell application \"System Events\" to get name of every application process"],
            capture_output=True, text=True, timeout=2
        )
        procs = r.stdout

        # Zoom/Teams/Webex/FaceTime: Prozess läuft = Call aktiv
        # Slack absichtlich ausgeschlossen — zu viele False Positives
        # (Slack-Calls manuell über Menü aufnehmen)
        for app, ctype in CALL_APPS.items():
            if app == "Slack":
                continue
            if app in procs:
                return app, ctype

        # Chrome: nur wenn Meet/Call-URL aktiv
        if "Google Chrome" in procs:
            url = get_chrome_url()
            if url and any(c in url for c in CALL_URLS):
                return "Google Chrome", "external"

    except Exception:
        pass
    return None, None

# ─── Meeting Recording ────────────────────────────────────
def _meeting_audio_callback(indata, frames, time_info, status):
    if meeting_active:
        meeting_frames.append(indata.copy())

def start_meeting_recording(app_name, call_type):
    global meeting_active, meeting_frames, meeting_start, meeting_stream, meeting_app, meeting_type
    with meeting_lock:
        if meeting_active:
            return
        meeting_active = True
        meeting_frames = []
        meeting_start  = datetime.now()
        meeting_app    = app_name
        meeting_type   = call_type
        meeting_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=_meeting_audio_callback, blocksize=1024
        )
        meeting_stream.start()

def stop_and_process_meeting():
    global meeting_active, meeting_stream
    with meeting_lock:
        meeting_active = False
        frames  = list(meeting_frames)
        t_start = meeting_start
        app     = meeting_app
        ctype   = meeting_type
        if meeting_stream:
            meeting_stream.stop()
            meeting_stream.close()
            meeting_stream = None

    if not frames:
        return

    try:
        duration    = (datetime.now() - t_start).total_seconds() if t_start else 0
        audio_data  = np.concatenate(frames, axis=0).flatten()
        audio_float = audio_data.astype(np.float32) / 32768.0

        subprocess.Popen(["osascript", "-e",
            'display notification "Transkribiere Meeting..." with title "whispr \u23f3"'])

        transcript = _transcribe_with_speakers(audio_float)
        summary    = _generate_meeting_summary(transcript, ctype, app)

        with get_db() as conn:
            conn.execute(
                "INSERT INTO meetings (app, call_type, transcript, summary, duration) VALUES (?,?,?,?,?)",
                (app, ctype, transcript, summary, round(duration, 1))
            )
            conn.commit()

        preview  = (summary or "Meeting gespeichert.")[:80]
        safe_pre = (preview
                    .replace('"', '\\"').replace("'", "\\'")
                    .replace("\u201c", '\\"').replace("\u201d", '\\"')
                    .replace("\u201e", '\\"').replace("\u2018", "\\'")
                    .replace("\u2019", "\\'"))
        subprocess.Popen(["osascript", "-e",
            f'display notification "{safe_pre}" with title "whispr \u2705 Meeting Summary"'])

    except Exception as e:
        subprocess.Popen(["osascript", "-e",
            f'display notification "Fehler: {str(e)[:80]}" with title "whispr \u26a0\ufe0f"'])

# ─── Speaker Diarization ──────────────────────────────────
_pyannote_pipeline = None

def _load_pyannote():
    global _pyannote_pipeline
    if _pyannote_pipeline is not None:
        return _pyannote_pipeline
    if not HF_TOKEN or not _PYANNOTE_AVAILABLE:
        return None
    try:
        _pyannote_pipeline = _PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("[pyannote] Pipeline geladen ✅")
        return _pyannote_pipeline
    except Exception as e:
        print(f"[pyannote] Nicht verfügbar: {e}")
        return None

def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def _transcribe_with_speakers(audio_float):
    """Transkribiert Audio. Mit pyannote: Speaker-Labels. Ohne: einfacher Text."""
    lang_param = None if LANGUAGE == "auto" else LANGUAGE
    segments, _ = model.transcribe(
        audio_float, language=lang_param, beam_size=5,
        vad_filter=VAD_FILTER, word_timestamps=False
    )
    segs = list(segments)

    pipeline = _load_pyannote()
    if pipeline is None or not _SOUNDFILE:
        # Kein Diarization — einfaches Transkript zurückgeben
        return "\n".join(f"[{_fmt_time(s.start)}] {s.text.strip()}" for s in segs)

    # Audio als .wav speichern für pyannote
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        _sf.write(tmp_path, audio_float, SAMPLE_RATE)
        diarization = pipeline(tmp_path)

        lines = []
        for seg in segs:
            seg_mid = (seg.start + seg.end) / 2
            speaker = "?"
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= seg_mid <= turn.end:
                    speaker = spk.replace("SPEAKER_", "Sprecher ")
                    break
            lines.append(f"[{speaker} – {_fmt_time(seg.start)}] {seg.text.strip()}")
        return "\n".join(lines)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ─── Testing Mode ────────────────────────────────────────
test_active              = False
test_session_name        = ""
test_start_time          = None
test_screenshots         = []   # list of {path, url, timestamp, notes:[]}
test_last_path           = None
test_lock                = threading.Lock()
_test_thread             = None
_test_session_dir        = None   # Fix: session_dir einmalig speichern

def _test_dir_for_session(name):
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.expanduser(f"~/.whispr-tests/{ts}_{safe}")
    os.makedirs(path, exist_ok=True)
    return path

def get_chrome_url() -> str:
    """Aktuelle URL aus Google Chrome holen."""
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "Google Chrome" to return URL of active tab of front window'],
            capture_output=True, text=True, timeout=2
        )
        return r.stdout.strip()
    except Exception:
        return ""

def _take_screenshot(path: str) -> bool:
    """Macht einen Screenshot des gesamten Bildschirms."""
    try:
        r = subprocess.run(["screencapture", "-x", "-o", path],
                           capture_output=True, timeout=5)
        if r.returncode != 0 or not os.path.exists(path):
            # Wahrscheinlich fehlende Screen Recording Permission
            print(f"[whispr] screencapture fehlgeschlagen: {r.stderr.decode()[:100]}")
            return False
        return True
    except Exception as e:
        print(f"[whispr] Screenshot-Fehler: {e}")
        return False

def _images_differ(path1: str, path2: str, threshold: float = 0.025) -> bool:
    """True wenn sich die Bilder um mehr als threshold unterscheiden."""
    if not _PIL:
        return True   # ohne Pillow: immer speichern
    try:
        img1 = np.array(_PILImage.open(path1).convert("RGB").resize((320, 200)))
        img2 = np.array(_PILImage.open(path2).convert("RGB").resize((320, 200)))
        diff = np.mean(np.abs(img1.astype(float) - img2.astype(float))) / 255.0
        return diff > threshold
    except Exception:
        return True

def _test_screenshot_loop(session_dir: str):
    """Background-Thread: Screenshot alle 1.5s, speichern wenn Screen sich ändert."""
    global test_last_path
    idx = 0
    while test_active:
        try:
            tmp = os.path.join(session_dir, f"_tmp_{int(time.monotonic()*1000)}.png")
            if _take_screenshot(tmp):
                if test_last_path is None or _images_differ(test_last_path, tmp):
                    url        = get_chrome_url()
                    final_path = os.path.join(session_dir, f"{idx:03d}_{datetime.now().strftime('%H%M%S')}.png")
                    os.rename(tmp, final_path)
                    with test_lock:
                        test_screenshots.append({
                            "path":      final_path,
                            "url":       url,
                            "timestamp": datetime.now().isoformat(),
                            "notes":     []
                        })
                    test_last_path = final_path
                    idx += 1
                else:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
        except Exception:
            pass
        time.sleep(1.5)

def _capture_note_anchor_screenshot():
    """Sofortiger Screenshot wenn Sprachnotiz beginnt — damit Notiz am richtigen Screen hängt."""
    global test_last_path
    if not _test_session_dir or not test_active:
        return
    try:
        ts    = datetime.now()
        path  = os.path.join(_test_session_dir, f"note_{ts.strftime('%H%M%S_%f')}.png")
        if _take_screenshot(path):
            url = get_chrome_url()
            with test_lock:
                # Nur hinzufügen wenn wirklich anderer Screen
                if test_last_path is None or _images_differ(test_last_path, path):
                    test_screenshots.append({
                        "path":        path,
                        "url":         url,
                        "timestamp":   ts.isoformat(),
                        "notes":       [],
                        "_note_anchor": True,  # Markierung: diese Notiz gehört hierhin
                    })
                    test_last_path = path
                else:
                    # Selber Screen — nur als Anchor an letzten Screenshot markieren
                    if test_screenshots:
                        test_screenshots[-1]["_note_anchor"] = True
                    if os.path.exists(path):
                        os.unlink(path)
    except Exception as e:
        print(f"[whispr] Note-Anchor-Screenshot fehlgeschlagen: {e}")

def start_test_session(name: str):
    global test_active, test_session_name, test_start_time, test_screenshots, test_last_path, _test_thread, _test_session_dir

    # Screen Recording Permission prüfen
    test_img = os.path.expanduser("~/.whispr-tests/_permission_check.png")
    os.makedirs(os.path.expanduser("~/.whispr-tests"), exist_ok=True)
    r = subprocess.run(["screencapture", "-x", "-o", test_img], capture_output=True, timeout=5)
    if r.returncode != 0 or not os.path.exists(test_img):
        subprocess.Popen(["osascript", "-e",
            'display alert "whispr: Bildschirmaufnahme fehlt" message '
            '"Bitte erlaube Bildschirmaufnahme unter:\\n'
            'Systemeinstellungen → Datenschutz → Bildschirmaufnahme → Terminal" '
            'buttons {"OK"}'])
        return
    os.unlink(test_img)

    session_dir           = _test_dir_for_session(name)
    _test_session_dir     = session_dir   # Fix: einmalig merken
    test_active           = True
    test_session_name     = name
    test_start_time       = datetime.now()
    test_screenshots      = []
    test_last_path        = None
    _test_thread = threading.Thread(target=_test_screenshot_loop, args=(session_dir,), daemon=True)
    _test_thread.start()

def stop_test_session(app_ref=None):
    global test_active
    if not test_active:
        return  # nichts zu stoppen
    test_active = False
    try:
        if _test_thread and _test_thread.is_alive():
            _test_thread.join(timeout=3)
    except Exception:
        pass

    with test_lock:
        shots    = list(test_screenshots)
        name     = test_session_name
        t_start  = test_start_time

    duration = (datetime.now() - t_start).total_seconds() if t_start else 0

    subprocess.Popen(["osascript", "-e",
        'display notification "Erstelle Notion-Report..." with title "whispr \U0001f9ea"'])

    threading.Thread(
        target=_process_and_export_test,
        args=(name, shots, duration),
        daemon=True
    ).start()

def _analyse_screenshot_with_claude(img_path: str, notes: list) -> str:
    """Schickt Screenshot + Notizen an Claude Vision → Beschreibung + Änderungsliste."""
    if not ANTHROPIC_API_KEY or not _ANTHROPIC or not os.path.exists(img_path):
        return "\n".join(notes) if notes else "(kein Kommentar)"
    try:
        import base64
        with open(img_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode()

        notes_txt = "\n".join(f"- {n}" for n in notes) if notes else "(keine Sprachnotizen)"
        prompt = (
            "Du analysierst einen Screenshot aus einem Produkt-Testing.\n\n"
            f"Sprachnotizen der Testerin zu diesem Screen:\n{notes_txt}\n\n"
            "Beschreibe in 1-2 Sätzen was auf dem Screen zu sehen ist (URL/Funktion). "
            "Liste dann die konkreten Änderungswünsche aus den Notizen strukturiert auf. "
            "Wenn keine Notizen: schreib 'Kein Feedback für diesen Screen.'\n"
            "Antworte auf Deutsch, kurz und actionable."
        )
        client = _anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": img_b64
                }},
                {"type": "text", "text": prompt}
            ]}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"(Vision-Analyse fehlgeschlagen: {e})\n" + "\n".join(notes)

def _process_and_export_test(name: str, shots: list, duration: float):
    """Analysiert alle Screenshots + Notizen → Notion-Page erstellen."""
    try:
        # Nur Screenshots mit Notizen ODER jeden 5. (für Kontext) analysieren
        analysed = []
        for i, s in enumerate(shots):
            has_notes = bool(s["notes"])
            if has_notes or i % 5 == 0:
                analysis = _analyse_screenshot_with_claude(s["path"], s["notes"])
                analysed.append({**s, "analysis": analysis, "has_notes": has_notes})

        notion_url = _create_notion_test_report(name, analysed, duration, len(shots))

        # Bug-Reports für jeden Screen mit Voice-Notes → direkt in Notion-Datenbank
        bug_count = 0
        for s in shots:
            if s.get("notes"):
                ok = _create_bug_report_entry(s["notes"], s.get("url", ""), name)
                if ok:
                    bug_count += 1

        # meta.json für Dashboard speichern (in das RICHTIGE Verzeichnis)
        import json as _json
        session_dir = _test_session_dir or _test_dir_for_session(name)
        meta = {
            "name": name,
            "screenshots": len(shots),
            "duration": duration,
            "notion_url": notion_url or "",
            "bug_reports": bug_count,
            "created_at": datetime.now().isoformat()
        }
        with open(os.path.join(session_dir, "meta.json"), "w") as f:
            _json.dump(meta, f)

        if notion_url or bug_count:
            n_fb = len([s for s in shots if s.get("notes")])
            subprocess.Popen(["osascript", "-e",
                f'display notification "Notion fertig \u2014 {n_fb} Bug-Reports erstellt" '
                f'with title "whispr \u2705 Test abgeschlossen"'])
        else:
            subprocess.Popen(["osascript", "-e",
                'display notification "Report lokal gespeichert (Notion-Token fehlt?)" '
                'with title "whispr \u26a0\ufe0f Test abgeschlossen"'])
    except Exception as e:
        subprocess.Popen(["osascript", "-e",
            f'display notification "Fehler: {str(e)[:80]}" with title "whispr \u26a0\ufe0f"'])

def _create_notion_test_report(name: str, shots: list, duration: float, total_screens: int):
    """Erstellt eine Notion-Page unter Testing & Feedback."""
    if not NOTION_TOKEN or not _REQUESTS:
        return None

    headers = {
        "Authorization":  f"Bearer {NOTION_TOKEN}",
        "Content-Type":   "application/json",
        "Notion-Version": "2022-06-28",
    }

    dur_min    = int(duration // 60)
    dur_sec    = int(duration % 60)
    date_str   = datetime.now().strftime("%d.%m.%Y %H:%M")
    n_feedback = len([s for s in shots if s.get("has_notes")])

    # Blocks bauen
    blocks = []

    # Header-Info
    blocks.append({"object": "block", "type": "callout", "callout": {
        "icon": {"type": "emoji", "emoji": "\U0001f9ea"},
        "color": "blue_background",
        "rich_text": [{"type": "text", "text": {"content":
            f"Session: {name}  |  {date_str}  |  Dauer: {dur_min}min {dur_sec}s  |  "
            f"{total_screens} Screens  |  {n_feedback} mit Feedback"
        }}]
    }})

    blocks.append({"object": "block", "type": "divider", "divider": {}})

    # Screens mit Feedback zuerst
    feedback_shots = [s for s in shots if s.get("has_notes")]
    if feedback_shots:
        blocks.append({"object": "block", "type": "heading_2", "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "\U0001f534 Screens mit Feedback"}}]
        }})
        for s in feedback_shots:
            url_txt = s.get("url", "") or "—"
            ts      = s.get("timestamp", "")[:16].replace("T", " ")
            blocks.append({"object": "block", "type": "heading_3", "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": f"\U0001f5bc\ufe0f {url_txt}"}}]
            }})
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": f"\U0001f552 {ts}"},
                               "annotations": {"color": "gray"}}]
            }})
            # Analyse (Claude Vision)
            for line in (s.get("analysis") or "").split("\n"):
                if line.strip():
                    blocks.append({"object": "block", "type": "paragraph", "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": line.strip()}}]
                    }})
            # Sprachnotizen
            if s["notes"]:
                blocks.append({"object": "block", "type": "callout", "callout": {
                    "icon": {"type": "emoji", "emoji": "\U0001f3a4"},
                    "color": "yellow_background",
                    "rich_text": [{"type": "text", "text": {"content":
                        "Sprachnotizen: " + " / ".join(s["notes"])
                    }}]
                }})
            blocks.append({"object": "block", "type": "divider", "divider": {}})

    # Alle anderen Screens
    other_shots = [s for s in shots if not s.get("has_notes")]
    if other_shots:
        blocks.append({"object": "block", "type": "heading_2", "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "\u2705 Screens ohne Feedback"}}]
        }})
        for s in other_shots:
            url_txt = s.get("url", "") or "—"
            ts      = s.get("timestamp", "")[:16].replace("T", " ")
            blocks.append({"object": "block", "type": "bulleted_list_item",
                           "bulleted_list_item": {"rich_text": [
                               {"type": "text", "text": {"content": f"{url_txt}"},
                                "annotations": {"code": True}},
                               {"type": "text", "text": {"content": f"  {ts}"},
                                "annotations": {"color": "gray"}}
                           ]}})

    # Seite erstellen (Notion: max 100 Blocks beim Create)
    payload = {
        "parent":     {"page_id": NOTION_TESTING_PAGE_ID},
        "icon":       {"type": "emoji", "emoji": "\U0001f9ea"},
        "properties": {"title": {"title": [{"type": "text", "text":
            {"content": f"\U0001f9ea {name} — {date_str}"}
        }]}},
        "children": blocks[:100]
    }

    r = _requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=15)
    if r.status_code != 200:
        return None

    page_id   = r.json().get("id")
    page_url  = r.json().get("url")

    # Restliche Blocks in Chunks anhängen
    remaining = blocks[100:]
    chunk_size = 100
    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i:i + chunk_size]
        _requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers,
            json={"children": chunk},
            timeout=15
        )

    return page_url

# ─── Bug Report → Notion Datenbank ──────────────────────
def _infer_area_from_url(url: str) -> str:
    """Leitet Area-Option aus der URL ab."""
    if not url:
        return "Other"
    url = url.lower()
    if "onboarding"  in url: return "Onboarding"
    if "questionnaire" in url or "fragebogen" in url: return "Questionnaire"
    if "notarization" in url or "notar" in url: return "Notarization"
    if "banking" in url or "konto" in url: return "Banking"
    if "document" in url or "vertrag" in url: return "Documents"
    if "support" in url or "chat" in url: return "Support Chat"
    if "dashboard" in url: return "Dashboard"
    if "payment" in url or "zahlung" in url or "checkout" in url: return "Payments"
    return "Other"

def _create_bug_report_entry(voice_notes: list, url: str, session_name: str) -> bool:
    """Erstellt einen Bug-Report-Eintrag in der Notion-Datenbank."""
    if not NOTION_TOKEN or not _REQUESTS:
        return False
    try:
        full_note = " / ".join(voice_notes)
        # Titel = erste 80 Zeichen der Notiz
        title = full_note[:80] + ("…" if len(full_note) > 80 else "")
        area  = _infer_area_from_url(url)
        steps = f"URL: {url}\nTest-Session: {session_name}" if url else f"Test-Session: {session_name}"

        headers = {
            "Authorization":  f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type":   "application/json",
        }
        payload = {
            "parent": {"database_id": NOTION_BUG_DB_ID},
            "properties": {
                "Title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                },
                "Actual Behavior": {
                    "rich_text": [{"type": "text", "text": {"content": full_note[:2000]}}]
                },
                "Steps to Reproduce": {
                    "rich_text": [{"type": "text", "text": {"content": steps[:2000]}}]
                },
                "Area": {
                    "select": {"name": area}
                },
                "Status": {
                    "select": {"name": "📥 New"}
                },
                "Type": {
                    "select": {"name": "🐛 Bug"}
                },
                "Reported By": {
                    "people": [{"id": MADELEINE_NOTION_ID}]
                },
            }
        }
        r = _requests.post(
            "https://api.notion.com/v1/pages",
            headers=headers, json=payload, timeout=10
        )
        return r.status_code == 200
    except Exception:
        return False

# ─── Meeting Summary (Claude API) ────────────────────────
def _generate_meeting_summary(transcript, call_type, app):
    if not ANTHROPIC_API_KEY or not _ANTHROPIC:
        return transcript[:600] + ("…" if len(transcript) > 600 else "")

    try:
        client = _anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
        if call_type == "internal":
            prompt = (
                f"Hier ist das Transkript eines internen Team-Calls via {app}.\n\n"
                f"Transkript:\n{transcript}\n\n"
                "Erstelle eine strukturierte Zusammenfassung auf Deutsch:\n"
                "- \U0001f4cb Kurzzusammenfassung (2-3 Sätze)\n"
                "- \u2705 Beschlossene Maßnahmen (mit Verantwortlichen wenn erkennbar)\n"
                "- \u2753 Offene Punkte / nächste Schritte\n\n"
                "Kurz und actionable."
            )
        else:
            prompt = (
                f"Hier ist das Transkript eines externen Calls via {app}.\n\n"
                f"Transkript:\n{transcript}\n\n"
                "Erstelle eine strukturierte Zusammenfassung auf Deutsch:\n"
                "- \U0001f4cb Kurzzusammenfassung (2-3 Sätze)\n"
                "- \U0001f91d Wichtigste besprochene Punkte\n"
                "- \u2705 Vereinbarte Maßnahmen / nächste Schritte\n"
                "- \u26a0\ufe0f Wichtige offene Punkte\n\n"
                "Kurz, professionell, actionable."
            )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"[Summary-Fehler: {e}]\n\nTranskript:\n{transcript[:400]}…"

def get_frontmost_app() -> str:
    """Gibt den Namen der aktiven App zurück (wo Text eingefügt wird)."""
    try:
        r = subprocess.run(
            ["osascript", "-e",
             "tell application \"System Events\" to get name of first "
             "application process whose frontmost is true"],
            capture_output=True, text=True, timeout=1
        )
        return r.stdout.strip()
    except Exception:
        return ""

def start_recording():
    global recording, audio_frames, record_start, stream
    with stream_lock:
        recording    = True
        audio_frames = []
        record_start = datetime.now()
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=1024,
        )
        stream.start()

def stop_and_transcribe(app_ref):
    global recording, stream
    with stream_lock:
        recording = False
        frames    = list(audio_frames)
        t_start   = record_start
        if stream is not None:
            stream.stop()
            stream.close()
            stream = None

    if not frames:
        if app_ref:
            app_ref.title = "🎙"
        return

    try:
        duration    = (datetime.now() - t_start).total_seconds() if t_start else 0
        audio_data  = np.concatenate(frames, axis=0).flatten()
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Dictionary words als initial_prompt → bessere Erkennung von Eigennamen
        with get_db() as conn:
            dict_words = conn.execute("SELECT word FROM dictionary").fetchall()
        initial_prompt = ", ".join(w["word"] for w in dict_words) if dict_words else None

        lang_param = None if LANGUAGE == "auto" else LANGUAGE
        segments, _ = model.transcribe(
            audio_float, language=lang_param, beam_size=5,
            vad_filter=VAD_FILTER, initial_prompt=initial_prompt
        )
        text = " ".join(seg.text for seg in segments).strip()

        # Apply snippets
        with get_db() as conn:
            snips = conn.execute("SELECT trigger, expansion FROM snippets").fetchall()
        for s in snips:
            text = text.replace(s["trigger"], s["expansion"])

        if text:
            global last_text
            last_text  = text
            word_count = len(text.split())
            wpm        = int(word_count / (duration / 60)) if duration > 0 else 0

            with get_db() as conn:
                conn.execute(
                    "INSERT INTO dictations (text, word_count, wpm, duration) VALUES (?,?,?,?)",
                    (text, word_count, wpm, duration)
                )
                conn.commit()

            # Im Test-Modus: Sprachnotiz dem aktuellen Screenshot anhängen
            if test_active:
                with test_lock:
                    if test_screenshots:
                        # Notiz dem Screenshot mit dem nächsten Timestamp anhängen
                        # (wurde beim Drücken von fn bereits aufgenommen — s. _attach_test_note_shot)
                        # Falls kein separater Screenshot: letzten nehmen
                        target = next(
                            (s for s in reversed(test_screenshots) if s.get("_note_anchor")),
                            test_screenshots[-1]
                        )
                        target["notes"].append(text)
                        # _note_anchor Flag zurücksetzen
                        target["_note_anchor"] = False

            subprocess.run(["pbcopy"], input=text.encode("utf-8"))
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down'
            ])

            # macOS Notification mit Vorschau des transkribierten Texts
            preview = text[:60] + ("…" if len(text) > 60 else "")
            safe    = (preview
                       .replace('"', '\\"').replace("'", "\\'")
                       .replace("\u201c", '\\"').replace("\u201d", '\\"')   # " "
                       .replace("\u201e", '\\"').replace("\u2018", "\\'")   # „ '
                       .replace("\u2019", "\\'")                            # '
                       )
            subprocess.Popen([
                "osascript", "-e",
                f'display notification "{safe}" with title "whispr ✅" subtitle "{word_count} Wörter · {wpm} WPM"'
            ])

    except Exception as e:
        subprocess.Popen([
            "osascript", "-e",
            f'display notification "Fehler: {str(e)[:80]}" with title "whispr ⚠️"'
        ])
    finally:
        if app_ref:
            app_ref.title = "🎙"
        overlay.hide()

# ─── Audio stream ─────────────────────────────────────────
def audio_callback(indata, frames, time, status):
    global _last_level_t
    if recording:
        audio_frames.append(indata.copy())
        # Level-Meter: RMS alle ~50 ms an Overlay schicken
        import time as _time
        now = _time.monotonic()
        if overlay and now - _last_level_t > 0.05:
            _last_level_t = now
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2))) / 32768.0
            overlay.update_level(min(rms * 14, 1.0))

# ─── Recording Overlay (AppKit / PyObjC — kein Tkinter) ──
if _APPKIT:
    class _LevelBarView(NSView):
        """Einfacher Fortschrittsbalken als NSView."""
        def initWithFrame_(self, frame):
            self = _objc.super(_LevelBarView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._level = 0.0
            return self

        def setLevel_(self, level: float):
            self._level = float(level)
            self.setNeedsDisplay_(True)

        def drawRect_(self, rect):
            bounds = self.bounds()
            # Hintergrund
            NSColor.colorWithRed_green_blue_alpha_(
                0.23, 0.23, 0.23, 1.0).setFill()
            NSBezierPath.fillRect_(bounds)
            # Balken
            w = bounds.size.width * min(max(self._level, 0.0), 1.0)
            NSColor.colorWithRed_green_blue_alpha_(
                0.39, 0.40, 0.95, 1.0).setFill()
            NSBezierPath.fillRect_(NSMakeRect(0, 0, w, bounds.size.height))


class RecordingOverlay:
    """
    Floating Always-on-top Fenster — gebaut mit AppKit/PyObjC.
    Kein Tkinter, kein Subprocess-Absturz.
    tick() muss vom Main-Thread aufgerufen werden (via rumps.Timer).
    """

    _W, _H = 310, 78

    def __init__(self):
        self._panel       = None
        self._status_fld  = None
        self._app_fld     = None
        self._bar_view    = None
        self._lock        = threading.Lock()
        self._cmd         = "hide"
        self._app_name    = ""
        self._lang        = "de"
        self._level       = 0.0
        self._smooth      = 0.0
        self._dirty       = False
        self._visible     = False
        self._shown_at    = None   # Zeitstempel wann overlay gezeigt wurde

        if not _APPKIT:
            return
        try:
            self._build_panel()
        except Exception as e:
            print(f"[overlay] AppKit-Setup fehlgeschlagen: {e}")
            self._panel = None

    def _build_panel(self):
        screen = NSScreen.mainScreen().frame()
        x = (screen.size.width - self._W) / 2
        y = 100  # px vom unteren Bildschirmrand (macOS: y=0 unten)

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, self._W, self._H),
            NSNonactivatingPanelMask | NSBorderlessWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setBackgroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.12, 0.96))
        panel.setOpaque_(False)
        panel.setHasShadow_(True)
        panel.orderOut_(None)   # initial versteckt

        cv = panel.contentView()
        pad = 18

        # Zeile 1: Status-Text
        status = NSTextField.alloc().initWithFrame_(
            NSMakeRect(pad, 44, self._W - pad * 2, 22))
        status.setStringValue_("🔴 Aufnahme läuft")
        status.setTextColor_(NSColor.whiteColor())
        status.setFont_(NSFont.boldSystemFontOfSize_(14))
        status.setEditable_(False)
        status.setBordered_(False)
        status.setBackgroundColor_(NSColor.clearColor())
        cv.addSubview_(status)

        # Zeile 2: Ziel-App
        app_fld = NSTextField.alloc().initWithFrame_(
            NSMakeRect(pad, 24, self._W - pad * 2, 16))
        app_fld.setStringValue_("")
        app_fld.setTextColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.55, 0.55, 0.58, 1.0))
        app_fld.setFont_(NSFont.systemFontOfSize_(11))
        app_fld.setEditable_(False)
        app_fld.setBordered_(False)
        app_fld.setBackgroundColor_(NSColor.clearColor())
        cv.addSubview_(app_fld)

        # Zeile 3: Audio-Level-Balken
        bar = _LevelBarView.alloc().initWithFrame_(
            NSMakeRect(pad, 10, self._W - pad * 2, 3))
        cv.addSubview_(bar)

        self._panel      = panel
        self._status_fld = status
        self._app_fld    = app_fld
        self._bar_view   = bar

    # ── Wird vom Main-Thread aufgerufen (rumps.Timer) ─────

    def tick(self, _=None):
        """50-ms-Tick auf dem Main-Thread — aktualisiert das Panel."""
        if not self._panel:
            return
        # Safety: Overlay nach 45s automatisch ausblenden
        with self._lock:
            if self._visible and self._shown_at and (time.monotonic() - self._shown_at) > 45:
                self._cmd      = "hide"
                self._shown_at = None
                self._dirty    = True
        with self._lock:
            if not self._dirty:
                # Level-Balken auch ohne dirty-Flag glätten
                if self._visible:
                    self._smooth *= 0.75
                    if self._bar_view:
                        self._bar_view.setLevel_(self._smooth)
                return
            cmd      = self._cmd
            app_name = self._app_name
            lang     = self._lang
            level    = self._level
            self._dirty = False

        if cmd == "show":
            txt = "🔴 Aufnahme läuft" if lang == "de" else "🔴 Recording"
            self._status_fld.setStringValue_(txt)
            self._status_fld.setTextColor_(NSColor.whiteColor())
            self._app_fld.setStringValue_(f"→ {app_name}" if app_name else "")
            self._smooth = 0.0
            if self._bar_view:
                self._bar_view.setLevel_(0.0)
            if not self._visible:
                self._panel.orderFront_(None)
                self._visible = True

        elif cmd == "transcribing":
            txt = "⏳ Transkribiere..." if lang == "de" else "⏳ Transcribing..."
            self._status_fld.setStringValue_(txt)
            self._status_fld.setTextColor_(
                NSColor.colorWithRed_green_blue_alpha_(0.68, 0.68, 0.7, 1.0))
            if self._bar_view:
                self._bar_view.setLevel_(0.0)

        elif cmd == "hide":
            if self._visible:
                self._panel.orderOut_(None)
                self._visible = False

        # Audio-Level-Balken glätten
        if self._visible and self._bar_view:
            self._smooth = self._smooth * 0.55 + level * 0.45
            self._bar_view.setLevel_(self._smooth)

    # ── Public API (thread-safe) ──────────────────────────

    def show(self, app_name: str = "", lang: str = "de"):
        with self._lock:
            self._cmd, self._app_name, self._lang = "show", app_name, lang
            self._shown_at = time.monotonic()
            self._dirty = True

    def set_transcribing(self, lang: str = "de"):
        with self._lock:
            self._cmd, self._lang = "transcribing", lang
            self._dirty = True

    def hide(self):
        with self._lock:
            self._cmd      = "hide"
            self._shown_at = None
            self._dirty    = True

    def update_level(self, raw_rms: float):
        with self._lock:
            self._level = min(raw_rms, 1.0)
            self._dirty = True


# ─── Dashboard HTML ───────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>whispr</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f7; color: #1d1d1f; display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  .sidebar { width: 200px; background: #fff; border-right: 1px solid #e5e5ea;
             padding: 24px 12px; display: flex; flex-direction: column; gap: 4px; flex-shrink: 0; }
  .sidebar-logo { font-size: 20px; font-weight: 700; padding: 0 12px 20px; color: #1d1d1f; }
  .nav-item { display: flex; align-items: center; gap: 10px; padding: 9px 12px;
              border-radius: 8px; cursor: pointer; font-size: 14px; color: #3a3a3c;
              transition: background 0.15s; user-select: none; }
  .nav-item:hover { background: #f2f2f7; }
  .nav-item.active { background: #f2f2f7; font-weight: 600; color: #1d1d1f; }
  .nav-icon { font-size: 16px; width: 20px; text-align: center; }
  .sidebar-bottom { margin-top: auto; padding: 12px; background: #f9f9fb;
                    border-radius: 10px; font-size: 12px; color: #6e6e73; }
  .sidebar-bottom strong { display: block; color: #1d1d1f; margin-bottom: 2px; }

  /* Main */
  .main { flex: 1; overflow-y: auto; padding: 32px; }
  h1 { font-size: 28px; font-weight: 700; margin-bottom: 24px; }
  h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; }

  /* Stats */
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px; }
  .stat-card { background: #fff; border-radius: 14px; padding: 20px 24px;
               border: 1px solid #e5e5ea; }
  .stat-value { font-size: 32px; font-weight: 700; color: #1d1d1f; }
  .stat-label { font-size: 13px; color: #6e6e73; margin-top: 4px; }

  /* Hero banner */
  .banner { background: #1d1d1f; border-radius: 14px; padding: 28px 32px;
            margin-bottom: 32px; color: #fff; display: flex; align-items: center;
            justify-content: space-between; }
  .banner-text h2 { color: #fff; font-size: 22px; }
  .banner-text p { color: #aeaeb2; font-size: 14px; margin-top: 6px; }
  .banner-icon { font-size: 48px; }

  /* History */
  .history-item { background: #fff; border-radius: 12px; padding: 16px 20px;
                  border: 1px solid #e5e5ea; margin-bottom: 10px;
                  display: flex; gap: 16px; align-items: flex-start; }
  .history-time { font-size: 12px; color: #6e6e73; white-space: nowrap; min-width: 50px; }
  .history-text { font-size: 14px; color: #1d1d1f; line-height: 1.5; }
  .history-meta { font-size: 12px; color: #aeaeb2; margin-top: 4px; }

  /* Form elements */
  .input-row { display: flex; gap: 10px; margin-bottom: 16px; }
  input[type=text] { flex: 1; padding: 9px 14px; border: 1px solid #d1d1d6;
                     border-radius: 8px; font-size: 14px; outline: none;
                     transition: border 0.15s; }
  input[type=text]:focus { border-color: #6366f1; }
  .btn { background: #6366f1; color: #fff; border: none; border-radius: 8px;
         padding: 9px 18px; font-size: 14px; cursor: pointer; font-weight: 500;
         transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn-sm { background: #f2f2f7; color: #6e6e73; border: none; border-radius: 6px;
            padding: 5px 10px; font-size: 12px; cursor: pointer; }
  .btn-sm:hover { background: #e5e5ea; }

  /* List items */
  .list-item { background: #fff; border: 1px solid #e5e5ea; border-radius: 10px;
               padding: 12px 16px; margin-bottom: 8px; display: flex;
               align-items: center; gap: 10px; }
  .trigger-tag { background: #eef2ff; color: #6366f1; padding: 3px 8px;
                 border-radius: 5px; font-size: 13px; font-weight: 600; }
  .arrow { color: #aeaeb2; }
  .expansion-text { flex: 1; font-size: 13px; color: #3a3a3c;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .del-btn { margin-left: auto; }
  .empty { color: #aeaeb2; font-size: 14px; padding: 20px 0; text-align: center; }

  /* Page sections */
  .page { display: none; }
  .page.active { display: block; }

  /* Settings */
  .setting-card { background: #fff; border: 1px solid #e5e5ea; border-radius: 14px;
                  padding: 20px 24px; margin-bottom: 16px; }
  .setting-card h3 { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
  .setting-card p  { font-size: 13px; color: #6e6e73; margin-bottom: 12px; }
  .setting-row { display: flex; align-items: center; justify-content: space-between; }
  kbd { background: #f2f2f7; border: 1px solid #d1d1d6; border-radius: 6px;
        padding: 3px 8px; font-size: 13px; font-family: -apple-system,sans-serif; }
  /* Toggle switch */
  .switch { position: relative; display: inline-block; width: 44px; height: 26px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; inset: 0; background: #d1d1d6;
            border-radius: 26px; transition: .2s; }
  .slider:before { position: absolute; content: ""; height: 20px; width: 20px;
                   left: 3px; bottom: 3px; background: #fff; border-radius: 50%;
                   transition: .2s; }
  input:checked + .slider { background: #6366f1; }
  input:checked + .slider:before { transform: translateX(18px); }
  /* Radio group */
  .radio-group { display: flex; gap: 12px; flex-wrap: wrap; }
  .radio-group label { display: flex; align-items: center; gap: 6px;
                       font-size: 14px; cursor: pointer; }
  /* Search bar */
  .history-header { display: flex; align-items: center;
                    justify-content: space-between; margin-bottom: 16px; }
  .history-header h2 { margin-bottom: 0; }
  .search-wrap { display: flex; gap: 8px; align-items: center; }
  input.search-input { width: 170px; padding: 7px 12px; border: 1px solid #d1d1d6;
                       border-radius: 8px; font-size: 13px; outline: none;
                       transition: border .15s; }
  input.search-input:focus { border-color: #6366f1; }
  .btn-danger { background: #ff3b30; color: #fff; border: none; border-radius: 8px;
                padding: 9px 18px; font-size: 14px; cursor: pointer; font-weight: 500;
                transition: opacity .15s; }
  .btn-danger:hover { opacity: .85; }
  .model-badge { display: inline-block; background: #eef2ff; color: #6366f1;
                 border-radius: 6px; padding: 3px 10px; font-size: 13px;
                 font-weight: 600; }
  /* Language toggle pill */
  .lang-toggle { display: flex; align-items: center; gap: 4px; margin: 0 12px 8px;
                 background: #f2f2f7; border-radius: 8px; padding: 4px; }
  .lang-btn { flex: 1; text-align: center; padding: 5px 4px; border-radius: 6px;
              font-size: 12px; cursor: pointer; border: none; background: transparent;
              color: #6e6e73; transition: background .15s, color .15s; }
  .lang-btn.active { background: #fff; color: #1d1d1f; font-weight: 600;
                     box-shadow: 0 1px 3px rgba(0,0,0,.1); }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-logo">🎙 whispr</div>
  <div class="nav-item active" onclick="nav('home')">
    <span class="nav-icon">🏠</span> Home
  </div>
  <div class="nav-item" onclick="nav('dictionary')">
    <span class="nav-icon">📖</span> Dictionary
  </div>
  <div class="nav-item" onclick="nav('snippets')">
    <span class="nav-icon">✂️</span> Snippets
  </div>
  <div class="nav-item" onclick="nav('meetings')">
    <span class="nav-icon">📋</span> <span id="nav-meetings">Meetings</span>
  </div>
  <div class="nav-item" onclick="nav('testing')">
    <span class="nav-icon">🧪</span> Testing
  </div>
  <div class="nav-item" onclick="nav('settings')">
    <span class="nav-icon">⚙️</span> <span id="nav-settings">Einstellungen</span>
  </div>
  <!-- Language toggle pill -->
  <div class="lang-toggle">
    <button class="lang-btn active" id="lang-btn-de" onclick="setUiLang('de')">🇩🇪 DE</button>
    <button class="lang-btn" id="lang-btn-en" onclick="setUiLang('en')">🇬🇧 EN</button>
    <button class="lang-btn" id="lang-btn-auto" onclick="setUiLang('auto')">🌐</button>
  </div>
  <div class="sidebar-bottom">
    <strong id="sidebar-hint-strong">⌥ rechts halten</strong>
    <span id="sidebar-hint">zum Diktieren</span>
  </div>
</div>

<div class="main">

  <!-- Home -->
  <div class="page active" id="page-home">
    <h1 id="h1-welcome">Willkommen zurück, Madeleine</h1>
    <div class="banner">
      <div class="banner-text">
        <h2 id="banner-heading">⌥ rechts halten → sprechen → loslassen</h2>
        <p id="banner-sub">Text wird direkt eingefügt — in jeder App.</p>
      </div>
      <div class="banner-icon">🎤</div>
    </div>
    <div class="stats">
      <div class="stat-card">
        <div class="stat-value" id="stat-words">—</div>
        <div class="stat-label" id="label-words">Wörter gesamt</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-wpm">—</div>
        <div class="stat-label" id="label-wpm">Ø WPM</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" id="stat-streak">—</div>
        <div class="stat-label" id="label-streak">Tage-Streak</div>
      </div>
    </div>
    <div class="history-header">
      <h2 id="h2-history">Verlauf</h2>
      <div class="search-wrap">
        <input class="search-input" id="history-search" type="text"
               placeholder="Suchen..." oninput="filterHistory(this.value)">
        <a href="/api/export/csv" class="btn-sm" id="btn-export" download>⬇ CSV</a>
      </div>
    </div>
    <div id="history-list"></div>
  </div>

  <!-- Dictionary -->
  <div class="page" id="page-dictionary">
    <h1 id="h1-dict">Dictionary</h1>
    <p id="dict-desc" style="color:#6e6e73;font-size:14px;margin-bottom:20px">
      Füge Namen und Fachbegriffe hinzu — z.B. RAKETENSTART, Supabase, Flamingo Innovations.
    </p>
    <div class="input-row">
      <input type="text" id="dict-input" placeholder="Begriff eingeben..." onkeydown="if(event.key==='Enter')addWord()">
      <button class="btn" id="btn-dict-add" onclick="addWord()">Hinzufügen</button>
    </div>
    <div id="word-list"></div>
  </div>

  <!-- Snippets -->
  <div class="page" id="page-snippets">
    <h1 id="h1-snip">Snippets</h1>
    <p id="snip-desc" style="color:#6e6e73;font-size:14px;margin-bottom:20px">
      Abkürzungen die beim Diktieren automatisch expandiert werden.<br>
      z.B. <strong>rs</strong> → <strong>RAKETENSTART</strong>
    </p>
    <div class="input-row">
      <input type="text" id="snip-trigger" placeholder="Abkürzung (z.B. rs)" style="max-width:140px" onkeydown="if(event.key==='Enter')addSnippet()">
      <span style="display:flex;align-items:center;color:#aeaeb2">→</span>
      <input type="text" id="snip-expansion" placeholder="Expansion (z.B. RAKETENSTART)" onkeydown="if(event.key==='Enter')addSnippet()">
      <button class="btn" id="btn-snip-add" onclick="addSnippet()">Hinzufügen</button>
    </div>
    <div id="snippet-list"></div>
  </div>

  <!-- Meetings -->
  <div class="page" id="page-meetings">
    <h1>📋 Meetings</h1>
    <div id="meeting-status-bar" style="display:none;background:#fef3c7;border:1px solid #f59e0b;border-radius:10px;padding:12px 16px;margin-bottom:20px;font-size:14px;"></div>
    <div id="meeting-list"><div class="empty">Noch keine Meeting-Aufnahmen.</div></div>
    <div id="meeting-detail" style="display:none;background:#fff;border:1px solid #e5e5ea;border-radius:14px;padding:24px;margin-top:16px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <h2 id="meeting-detail-title" style="margin:0"></h2>
        <button class="btn-sm" onclick="closeMeetingDetail()">✕ Schließen</button>
      </div>
      <h3 style="font-size:14px;font-weight:600;margin-bottom:8px">📋 Summary</h3>
      <div id="meeting-summary" style="font-size:14px;line-height:1.7;white-space:pre-wrap;margin-bottom:20px"></div>
      <h3 style="font-size:14px;font-weight:600;margin-bottom:8px">📝 Transkript</h3>
      <div id="meeting-transcript" style="font-size:12px;color:#6e6e73;line-height:1.7;white-space:pre-wrap;background:#f9f9fb;border-radius:8px;padding:12px"></div>
    </div>
  </div>

  <!-- Settings -->
  <div class="page" id="page-settings">
    <h1 id="h1-settings">Einstellungen</h1>

    <!-- Language -->
    <div class="setting-card">
      <h3 id="set-h-lang">Erkennungssprache</h3>
      <p id="set-p-lang">DE = Deutsch, EN = Englisch, Auto = automatisch erkennen (etwas langsamer).</p>
      <div class="radio-group" id="lang-radio-group">
        <label><input type="radio" name="set-lang" value="de" onchange="saveLang(this.value)"> 🇩🇪 Deutsch</label>
        <label><input type="radio" name="set-lang" value="en" onchange="saveLang(this.value)"> 🇬🇧 English</label>
        <label><input type="radio" name="set-lang" value="auto" onchange="saveLang(this.value)"> 🌐 Auto-Detect</label>
      </div>
    </div>

    <!-- VAD -->
    <div class="setting-card">
      <div class="setting-row">
        <div>
          <h3 id="set-h-vad">Voice Activity Detection</h3>
          <p id="set-p-vad" style="margin-bottom:0">Filtert automatisch Stille aus der Aufnahme.</p>
        </div>
        <label class="switch">
          <input type="checkbox" id="vad-toggle" onchange="saveVad(this.checked)">
          <span class="slider"></span>
        </label>
      </div>
    </div>

    <!-- Model -->
    <div class="setting-card">
      <h3 id="set-h-model">Whisper-Modell</h3>
      <p id="set-p-model">Aktuell aktiv: <span class="model-badge" id="setting-model">small</span> — tiny ist schneller, medium ist genauer. Wechsel erfordert App-Neustart (MODEL_SIZE in whispr.py).</p>
    </div>

    <!-- Hotkey -->
    <div class="setting-card">
      <h3 id="set-h-hotkey">Tastenkürzel</h3>
      <p id="set-p-hotkey"><kbd>⌥ Rechts</kbd> gedrückt halten → sprechen → loslassen (Hold-Modus)<br>oder im Menü auf Toggle-Modus wechseln.</p>
    </div>

    <!-- Clear history -->
    <div class="setting-card">
      <h3 id="set-h-clear">Verlauf löschen</h3>
      <p id="set-p-clear">Löscht alle gespeicherten Diktate. Wörter & WPM-Statistiken werden zurückgesetzt.</p>
      <button class="btn-danger" id="btn-clear" onclick="clearHistory()">🗑 Verlauf löschen</button>
    </div>
  </div>

  <!-- Testing -->
  <div class="page" id="page-testing">
    <h1>🧪 Testing</h1>
    <div style="margin-bottom:20px;display:flex;gap:10px;align-items:center">
      <button onclick="startTest()" id="btn-test-start" style="background:#1d1d1f;color:#fff;border:none;padding:10px 20px;border-radius:8px;font-size:14px;cursor:pointer">▶ Test starten</button>
      <button onclick="stopTest()" id="btn-test-stop" style="background:#ff3b30;color:#fff;border:none;padding:10px 20px;border-radius:8px;font-size:14px;cursor:pointer;display:none">⏹ Test stoppen & Notion-Report</button>
      <span id="test-status" style="font-size:13px;color:#666"></span>
    </div>
    <div id="test-session-list"><div class="empty">Noch keine Test-Sessions.</div></div>
  </div>

</div>

<script>
let _page = 'home';
let _lang = 'de';

const T = {
  de: {
    welcome:       'Willkommen zurück, Madeleine',
    banner_h:      '⌥ rechts halten → sprechen → loslassen',
    banner_p:      'Text wird direkt eingefügt — in jeder App.',
    words:         'Wörter gesamt',
    wpm:           'Ø WPM',
    streak:        'Tage-Streak',
    history:       'Verlauf',
    hint_strong:   '⌥ rechts halten',
    hint:          'zum Diktieren',
    h1_dict:       'Dictionary',
    dict_desc:     'Füge Namen und Fachbegriffe hinzu — z.B. RAKETENSTART, Supabase, Flamingo Innovations.',
    dict_ph:       'Begriff eingeben...',
    dict_add:      'Hinzufügen',
    dict_del:      'Löschen',
    dict_empty:    'Noch keine Einträge.',
    h1_snip:       'Snippets',
    snip_desc:     'Abkürzungen die beim Diktieren automatisch expandiert werden.',
    snip_trig_ph:  'Abkürzung (z.B. rs)',
    snip_exp_ph:   'Expansion (z.B. RAKETENSTART)',
    snip_add:      'Hinzufügen',
    snip_del:      'Löschen',
    snip_empty:    'Noch keine Snippets.',
    empty_hist:    'Noch keine Diktate. Los geht&#39;s! 🎙',
    words_label:   'Wörter',
    search_ph:     'Suchen...',
    nav_settings:  'Einstellungen',
    h1_settings:   'Einstellungen',
    set_h_lang:    'Erkennungssprache',
    set_p_lang:    'DE = Deutsch, EN = Englisch, Auto = automatisch erkennen.',
    set_h_vad:     'Voice Activity Detection',
    set_p_vad:     'Filtert automatisch Stille aus der Aufnahme.',
    set_h_model:   'Whisper-Modell',
    set_p_model:   'Aktiv: tiny = schnell, small = Standard, medium = genauer. Wechsel erfordert Neustart.',
    set_h_hotkey:  'Tastenkürzel',
    set_h_clear:   'Verlauf löschen',
    set_p_clear:   'Löscht alle gespeicherten Diktate. Statistiken werden zurückgesetzt.',
    btn_clear:     '🗑 Verlauf löschen',
    clear_confirm: 'Wirklich alle Diktate löschen?',
  },
  en: {
    welcome:       'Welcome back, Madeleine',
    banner_h:      'Hold ⌥ right → speak → release',
    banner_p:      'Text is inserted directly — in any app.',
    words:         'Total words',
    wpm:           'Avg WPM',
    streak:        'Day streak',
    history:       'History',
    hint_strong:   'Hold ⌥ right',
    hint:          'to dictate',
    h1_dict:       'Dictionary',
    dict_desc:     'Add names and technical terms — e.g. RAKETENSTART, Supabase, Flamingo Innovations.',
    dict_ph:       'Enter term...',
    dict_add:      'Add',
    dict_del:      'Delete',
    dict_empty:    'No entries yet.',
    h1_snip:       'Snippets',
    snip_desc:     'Abbreviations that auto-expand during dictation.',
    snip_trig_ph:  'Abbreviation (e.g. rs)',
    snip_exp_ph:   'Expansion (e.g. RAKETENSTART)',
    snip_add:      'Add',
    snip_del:      'Delete',
    snip_empty:    'No snippets yet.',
    empty_hist:    'No dictations yet. Let&#39;s go! 🎙',
    words_label:   'words',
    search_ph:     'Search...',
    nav_settings:  'Settings',
    h1_settings:   'Settings',
    set_h_lang:    'Recognition language',
    set_p_lang:    'DE = German, EN = English, Auto = auto-detect (slightly slower).',
    set_h_vad:     'Voice Activity Detection',
    set_p_vad:     'Automatically filters silence from recordings.',
    set_h_model:   'Whisper model',
    set_p_model:   'Active: tiny = fast, small = standard, medium = more accurate. Change requires restart.',
    set_h_hotkey:  'Keyboard shortcut',
    set_h_clear:   'Clear history',
    set_p_clear:   'Deletes all saved dictations. Stats reset to zero.',
    btn_clear:     '🗑 Clear history',
    clear_confirm: 'Really delete all dictations?',
  }
};

function _updateLangPill(lang) {
  ['de','en','auto'].forEach(l => {
    const btn = document.getElementById('lang-btn-'+l);
    if (btn) btn.classList.toggle('active', l === lang);
  });
}

function applyLang(lang) {
  if (lang !== 'de' && lang !== 'en' && lang !== 'auto') return;
  // For 'auto': keep current UI language, just mark pill
  const tLang = (lang === 'auto') ? (_lang || 'de') : lang;
  if (lang !== 'auto') _lang = lang;
  _updateLangPill(lang);
  const t = T[tLang] || T['de'];
  document.getElementById('h1-welcome').textContent      = t.welcome;
  document.getElementById('banner-heading').textContent  = t.banner_h;
  document.getElementById('banner-sub').textContent      = t.banner_p;
  document.getElementById('label-words').textContent     = t.words;
  document.getElementById('label-wpm').textContent       = t.wpm;
  document.getElementById('label-streak').textContent    = t.streak;
  document.getElementById('h2-history').textContent      = t.history;
  document.getElementById('sidebar-hint-strong').textContent = t.hint_strong;
  document.getElementById('sidebar-hint').textContent    = t.hint;
  document.getElementById('h1-dict').textContent         = t.h1_dict;
  document.getElementById('dict-desc').textContent       = t.dict_desc;
  document.getElementById('dict-input').placeholder      = t.dict_ph;
  document.getElementById('btn-dict-add').textContent    = t.dict_add;
  document.getElementById('h1-snip').textContent         = t.h1_snip;
  document.getElementById('snip-desc').textContent       = t.snip_desc;
  document.getElementById('snip-trigger').placeholder    = t.snip_trig_ph;
  document.getElementById('snip-expansion').placeholder  = t.snip_exp_ph;
  document.getElementById('btn-snip-add').textContent    = t.snip_add;
  document.getElementById('history-search').placeholder  = t.search_ph;
  document.getElementById('nav-settings').textContent    = t.nav_settings;
  document.getElementById('h1-settings').textContent     = t.h1_settings;
  document.getElementById('set-h-lang').textContent      = t.set_h_lang;
  document.getElementById('set-p-lang').textContent      = t.set_p_lang;
  document.getElementById('set-h-vad').textContent       = t.set_h_vad;
  document.getElementById('set-p-vad').textContent       = t.set_p_vad;
  document.getElementById('set-h-model').textContent     = t.set_h_model;
  document.getElementById('set-h-hotkey').textContent    = t.set_h_hotkey;
  document.getElementById('set-h-clear').textContent     = t.set_h_clear;
  document.getElementById('set-p-clear').textContent     = t.set_p_clear;
  document.getElementById('btn-clear').textContent       = t.btn_clear;
}

async function setUiLang(lang) {
  // Save to backend (which also updates the menu bar app)
  await fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({language: lang})
  });
  applyLang(lang);
  // Sync settings radio
  const radio = document.querySelector(`input[name="set-lang"][value="${lang}"]`);
  if (radio) radio.checked = true;
  // Re-render current page content in new language
  if (_page === 'home')       loadHomeData();
  if (_page === 'dictionary') loadDictionary();
  if (_page === 'snippets')   loadSnippets();
}

async function checkLang() {
  try {
    const data = await fetch('/api/lang').then(r => r.json());
    const serverLang = data.lang;
    // Sync pill regardless of whether UI lang changed
    _updateLangPill(serverLang);
    const uiLang = serverLang === 'auto' ? _lang : serverLang;
    if (uiLang !== _lang) applyLang(serverLang);
    // Sync settings radio button
    const radio = document.querySelector(`input[name="set-lang"][value="${serverLang}"]`);
    if (radio) radio.checked = true;
  } catch(e) {}
}

function nav(name) {
  const pages = ['home','dictionary','snippets','meetings','settings'];
  document.querySelectorAll('.nav-item').forEach((el,i) => {
    el.classList.toggle('active', pages[i] === name);
  });
  document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  _page = name;
  if (name === 'home')       loadHome();
  if (name === 'dictionary') loadDictionary();
  if (name === 'snippets')   loadSnippets();
  if (name === 'meetings')   loadMeetings();
  if (name === 'settings')   loadSettings();
  if (name === 'testing')    loadTesting();
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function loadHome() {
  await loadHomeData();
}

async function loadDictionary() {
  const words = await fetch('/api/dictionary').then(r=>r.json());
  const el = document.getElementById('word-list');
  if (!words.length) { el.innerHTML='<div class="empty">'+T[_lang].dict_empty+'</div>'; return; }
  el.innerHTML = words.map(w=>`
    <div class="list-item">
      <span style="font-size:14px;flex:1">${esc(w.word)}</span>
      <button class="btn-sm del-btn" onclick="deleteWord(${w.id})">${T[_lang].dict_del}</button>
    </div>`).join('');
}

async function addWord() {
  const input = document.getElementById('dict-input');
  const word  = input.value.trim();
  if (!word) return;
  await fetch('/api/dictionary',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({word})});
  input.value = '';
  loadDictionary();
}

async function deleteWord(id) {
  await fetch('/api/dictionary/'+id,{method:'DELETE'});
  loadDictionary();
}

async function loadSnippets() {
  const snips = await fetch('/api/snippets').then(r=>r.json());
  const el = document.getElementById('snippet-list');
  if (!snips.length) { el.innerHTML='<div class="empty">'+T[_lang].snip_empty+'</div>'; return; }
  el.innerHTML = snips.map(s=>`
    <div class="list-item">
      <span class="trigger-tag">${esc(s.trigger)}</span>
      <span class="arrow">→</span>
      <span class="expansion-text">${esc(s.expansion)}</span>
      <button class="btn-sm del-btn" onclick="deleteSnippet(${s.id})">${T[_lang].snip_del}</button>
    </div>`).join('');
}

async function addSnippet() {
  const trigger   = document.getElementById('snip-trigger').value.trim();
  const expansion = document.getElementById('snip-expansion').value.trim();
  if (!trigger || !expansion) return;
  await fetch('/api/snippets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trigger,expansion})});
  document.getElementById('snip-trigger').value   = '';
  document.getElementById('snip-expansion').value = '';
  loadSnippets();
}

async function deleteSnippet(id) {
  await fetch('/api/snippets/'+id,{method:'DELETE'});
  loadSnippets();
}

// ── History search ───────────────────────────────────────
let _allHistory = [];
async function loadHomeData() {
  const [stats, history] = await Promise.all([
    fetch('/api/stats').then(r=>r.json()),
    fetch('/api/history').then(r=>r.json()),
  ]);
  const locale = _lang === 'de' ? 'de' : 'en';
  document.getElementById('stat-words').textContent  = Number(stats.total_words).toLocaleString(locale);
  document.getElementById('stat-wpm').textContent    = stats.avg_wpm || '—';
  document.getElementById('stat-streak').textContent = stats.streak;
  _allHistory = history;
  renderHistory(_allHistory);
}

function renderHistory(items) {
  const locale = _lang === 'de' ? 'de' : 'en';
  const el = document.getElementById('history-list');
  if (!items.length) {
    el.innerHTML = '<div class="empty">' + T[_lang].empty_hist + '</div>';
    return;
  }
  el.innerHTML = items.map(d => {
    const dt   = new Date(d.created_at);
    const time = dt.toLocaleTimeString(locale,{hour:'2-digit',minute:'2-digit'});
    const day  = dt.toLocaleDateString(locale,{day:'2-digit',month:'short'});
    return `<div class="history-item">
      <div class="history-time">${time}<br><small>${day}</small></div>
      <div style="flex:1">
        <div class="history-text">${esc(d.text)}</div>
        <div class="history-meta">${d.word_count} ${T[_lang].words_label} · ${d.wpm} WPM</div>
      </div>
    </div>`;
  }).join('');
}

function filterHistory(q) {
  const term = q.toLowerCase().trim();
  if (!term) { renderHistory(_allHistory); return; }
  renderHistory(_allHistory.filter(d => d.text.toLowerCase().includes(term)));
}

// ── Settings ─────────────────────────────────────────────
async function loadSettings() {
  try {
    const s = await fetch('/api/settings').then(r=>r.json());
    document.getElementById('setting-model').textContent = s.model;
    document.getElementById('vad-toggle').checked = !!s.vad;
    const radio = document.querySelector(`input[name="set-lang"][value="${s.language}"]`);
    if (radio) radio.checked = true;
  } catch(e) {}
}

async function saveLang(lang) {
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({language:lang})});
}

async function saveVad(checked) {
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({vad:checked})});
}

async function clearHistory() {
  const t = T[_lang] || T['de'];
  if (!confirm(t.clear_confirm)) return;
  await fetch('/api/history/clear',{method:'POST'});
  _allHistory = [];
  renderHistory([]);
  // Refresh stats
  const stats = await fetch('/api/stats').then(r=>r.json());
  document.getElementById('stat-words').textContent  = '0';
  document.getElementById('stat-wpm').textContent    = '—';
  document.getElementById('stat-streak').textContent = '0';
}

// ── Meetings ─────────────────────────────────────────────
async function loadMeetings() {
  const meetings = await fetch('/api/meetings').then(r=>r.json());
  const el = document.getElementById('meeting-list');
  if (!meetings.length) { el.innerHTML='<div class="empty">Noch keine Meeting-Aufnahmen.</div>'; return; }
  el.innerHTML = meetings.map(m => {
    const dt    = new Date(m.created_at);
    const time  = dt.toLocaleTimeString('de',{hour:'2-digit',minute:'2-digit'});
    const day   = dt.toLocaleDateString('de',{day:'2-digit',month:'short'});
    const icon  = m.call_type === 'internal' ? '👥' : '📞';
    const label = m.call_type === 'internal' ? 'Team-Call' : 'Externer Call';
    const dur   = m.duration ? Math.round(m.duration/60)+'min' : '';
    const preview = (m.summary || '').slice(0,120).split('\\n').join(' ');
    return `<div class="history-item" style="cursor:pointer" onclick="openMeeting(${m.id})">
      <div class="history-time">${time}<br><small>${day}</small></div>
      <div style="flex:1">
        <div style="font-size:13px;font-weight:600;margin-bottom:4px">${icon} ${label} · ${esc(m.app||'')} ${dur ? '· '+dur : ''}</div>
        <div class="history-text">${esc(preview)}${preview.length>=120?'…':''}</div>
      </div>
    </div>`;
  }).join('');

  // Meeting status bar
  const status = await fetch('/api/meetings/status').then(r=>r.json());
  const bar = document.getElementById('meeting-status-bar');
  if (status.active) {
    const icon  = status.call_type === 'internal' ? '👥' : '📞';
    bar.style.display = 'block';
    bar.textContent   = icon + ' Aufnahme läuft — ' + (status.app||'') + '. Stoppe das Meeting über das Menü-Icon.';
  } else {
    bar.style.display = 'none';
  }
}

async function openMeeting(id) {
  const m = await fetch('/api/meetings/'+id).then(r=>r.json());
  const icon  = m.call_type === 'internal' ? '👥' : '📞';
  const label = m.call_type === 'internal' ? 'Team-Call' : 'Externer Call';
  const dt    = new Date(m.created_at).toLocaleString('de');
  document.getElementById('meeting-detail-title').textContent = icon+' '+label+' · '+dt;
  document.getElementById('meeting-summary').textContent    = m.summary || '—';
  document.getElementById('meeting-transcript').textContent = m.transcript || '—';
  document.getElementById('meeting-detail').style.display   = 'block';
  document.getElementById('meeting-detail').scrollIntoView({behavior:'smooth'});
}

function closeMeetingDetail() {
  document.getElementById('meeting-detail').style.display = 'none';
}

let _pollInterval = null;
function startPolling() {
  if (_pollInterval) return;
  _pollInterval = setInterval(async () => {
    await checkLang();
    if (_page === 'home'     && !document.hidden) loadHomeData();
    if (_page === 'meetings' && !document.hidden) loadMeetings();
  }, 3000);
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { clearInterval(_pollInterval); _pollInterval = null; }
  else startPolling();
});
checkLang().then(() => { startPolling(); loadHome(); });

// ── Testing ──────────────────────────────────────────────
async function loadTesting() {
  const status = await fetch('/api/test/status').then(r=>r.json()).catch(()=>({active:false}));
  const start = document.getElementById('btn-test-start');
  const stop  = document.getElementById('btn-test-stop');
  const stat  = document.getElementById('test-status');
  if (status.active) {
    start.style.display = 'none';
    stop.style.display  = 'inline-block';
    stat.textContent    = 'Läuft: ' + (status.name || '') + ' …';
  } else {
    start.style.display = 'inline-block';
    stop.style.display  = 'none';
    stat.textContent    = '';
  }
  const sessions = await fetch('/api/test/sessions').then(r=>r.json()).catch(()=>[]);
  const el = document.getElementById('test-session-list');
  if (!sessions.length) { el.innerHTML = '<div class="empty">Noch keine Test-Sessions.</div>'; return; }
  el.innerHTML = sessions.map(s => {
    const dt  = new Date(s.created_at);
    const day = dt.toLocaleDateString('de',{day:'2-digit',month:'short',year:'numeric'});
    const tim = dt.toLocaleTimeString('de',{hour:'2-digit',minute:'2-digit'});
    return '<div class="history-item"><div class="history-time">'+tim+'<br><small>'+day+'</small></div>'
      +'<div style="flex:1"><div style="font-weight:600;font-size:13px;margin-bottom:4px">'+esc(s.name)+'</div>'
      +'<div class="history-text">'+s.screenshots+' Screenshots · '+Math.round(s.duration/60)+' min'
      +(s.notion_url ? ' · <a href="'+s.notion_url+'" target="_blank" style="color:#007aff">Notion Report</a>' : '')
      +'</div></div></div>';
  }).join('');
}

async function startTest() {
  const name = prompt('Test-Session Name:', 'GmbH Flow v1');
  if (!name) return;
  await fetch('/api/test/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})});
  loadTesting();
}

async function stopTest() {
  document.getElementById('test-status').textContent = 'Erstelle Notion-Report …';
  await fetch('/api/test/stop', {method:'POST'});
  setTimeout(loadTesting, 2000);
}
</script>
</body>
</html>"""

# ─── Overlay initialisieren ───────────────────────────────
overlay = RecordingOverlay()

# ─── Flask API ────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return Response(DASHBOARD_HTML, mimetype="text/html")

@flask_app.route("/api/stats")
def api_stats():
    with get_db() as conn:
        total_words = conn.execute("SELECT COALESCE(SUM(word_count),0) FROM dictations").fetchone()[0]
        avg_wpm     = conn.execute("SELECT COALESCE(ROUND(AVG(wpm),0),0) FROM dictations WHERE wpm>0").fetchone()[0]
        rows        = conn.execute("SELECT DATE(created_at) as day FROM dictations GROUP BY day ORDER BY day DESC").fetchall()
    streak = 0
    today  = date.today()
    for i, row in enumerate(rows):
        try:
            d = datetime.strptime(row["day"], "%Y-%m-%d").date()
        except Exception:
            break
        # i==0: heute (0 Tage) ODER gestern (1 Tag) zählt als Streak-Start
        expected = i if (rows and datetime.strptime(rows[0]["day"], "%Y-%m-%d").date() == today) else i + 1
        if (today - d).days == expected:
            streak += 1
        else:
            break
    return jsonify(total_words=total_words, avg_wpm=int(avg_wpm), streak=streak)

@flask_app.route("/api/history")
def api_history():
    q = request.args.get("q", "").strip()
    with get_db() as conn:
        if q:
            rows = conn.execute(
                "SELECT id,text,word_count,wpm,created_at FROM dictations "
                "WHERE text LIKE ? ORDER BY created_at DESC LIMIT 100",
                (f"%{q}%",)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id,text,word_count,wpm,created_at FROM dictations "
                "ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
    return jsonify([dict(r) for r in rows])

@flask_app.route("/api/dictionary", methods=["GET"])
def api_dict_get():
    with get_db() as conn:
        rows = conn.execute("SELECT id,word FROM dictionary ORDER BY word").fetchall()
    return jsonify([dict(r) for r in rows])

@flask_app.route("/api/dictionary", methods=["POST"])
def api_dict_add():
    word = (request.json or {}).get("word","").strip()
    if not word: return jsonify(error="empty"), 400
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO dictionary (word) VALUES (?)", (word,))
            conn.commit()
        return jsonify(ok=True)
    except sqlite3.IntegrityError:
        return jsonify(error="exists"), 409

@flask_app.route("/api/dictionary/<int:wid>", methods=["DELETE"])
def api_dict_del(wid):
    with get_db() as conn:
        conn.execute("DELETE FROM dictionary WHERE id=?", (wid,))
        conn.commit()
    return jsonify(ok=True)

@flask_app.route("/api/snippets", methods=["GET"])
def api_snip_get():
    with get_db() as conn:
        rows = conn.execute("SELECT id,trigger,expansion FROM snippets ORDER BY trigger").fetchall()
    return jsonify([dict(r) for r in rows])

@flask_app.route("/api/snippets", methods=["POST"])
def api_snip_add():
    data      = request.json or {}
    trigger   = data.get("trigger","").strip()
    expansion = data.get("expansion","").strip()
    if not trigger or not expansion: return jsonify(error="empty"), 400
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO snippets (trigger,expansion) VALUES (?,?)", (trigger, expansion))
            conn.commit()
        return jsonify(ok=True)
    except sqlite3.IntegrityError:
        return jsonify(error="exists"), 409

@flask_app.route("/api/snippets/<int:sid>", methods=["DELETE"])
def api_snip_del(sid):
    with get_db() as conn:
        conn.execute("DELETE FROM snippets WHERE id=?", (sid,))
        conn.commit()
    return jsonify(ok=True)

@flask_app.route("/api/lang")
def api_lang():
    return jsonify(lang=LANGUAGE)

@flask_app.route("/api/export/csv")
def api_export_csv():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,text,word_count,wpm,duration,created_at "
            "FROM dictations ORDER BY created_at DESC"
        ).fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "text", "word_count", "wpm", "duration_s", "created_at"])
    for r in rows:
        w.writerow([r["id"], r["text"], r["word_count"], r["wpm"],
                    round(r["duration"], 1), r["created_at"]])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=whispr_history.csv"}
    )

@flask_app.route("/api/settings", methods=["GET"])
def api_settings_get():
    return jsonify(
        model    = MODEL_SIZE,
        language = LANGUAGE,
        vad      = VAD_FILTER,
        hotkey   = "⌥ Rechts",
    )

@flask_app.route("/api/settings", methods=["POST"])
def api_settings_post():
    global VAD_FILTER, LANGUAGE
    data = request.json or {}
    if "vad" in data:
        VAD_FILTER = bool(data["vad"])
        set_setting("vad_filter", "1" if VAD_FILTER else "0")
    if "language" in data and data["language"] in ("de", "en", "auto"):
        LANGUAGE = data["language"]
        set_setting("language", LANGUAGE)
    return jsonify(ok=True, model=MODEL_SIZE, language=LANGUAGE, vad=VAD_FILTER)

@flask_app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    with get_db() as conn:
        conn.execute("DELETE FROM dictations")
        conn.commit()
    return jsonify(ok=True)

@flask_app.route("/api/meetings")
def api_meetings():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, app, call_type, summary, duration, created_at "
            "FROM meetings ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@flask_app.route("/api/meetings/<int:mid>")
def api_meeting_detail(mid):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id=?", (mid,)).fetchone()
    if not row:
        return jsonify(error="not found"), 404
    return jsonify(dict(row))

@flask_app.route("/api/meetings/status")
def api_meeting_status():
    return jsonify(active=meeting_active, app=meeting_app, call_type=meeting_type)

@flask_app.route("/api/test/status")
def api_test_status():
    return jsonify(active=test_active, name=test_session_name)

@flask_app.route("/api/test/start", methods=["POST"])
def api_test_start():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "Test")
    start_test_session(name)
    return jsonify(ok=True)

@flask_app.route("/api/test/stop", methods=["POST"])
def api_test_stop():
    threading.Thread(target=stop_test_session, daemon=True).start()
    return jsonify(ok=True)

@flask_app.route("/api/test/sessions")
def api_test_sessions():
    try:
        import glob, json as _json
        base = os.path.expanduser("~/.whispr-tests")
        sessions = []
        for d in sorted(glob.glob(os.path.join(base, "*")), reverse=True)[:20]:
            meta_path = os.path.join(d, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    sessions.append(_json.load(f))
        return jsonify(sessions)
    except Exception:
        return jsonify([])

def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(port=PORT, debug=False, use_reloader=False)

# ─── Menu Bar App ─────────────────────────────────────────
class WhisprApp(rumps.App):
    def __init__(self):
        super().__init__("🎙", quit_button=None)
        self._call_detected_app  = None
        self._call_detected_type = None
        self._test_running       = False
        self._setup_menu()
        threading.Thread(target=run_flask, daemon=True).start()
        self._check_accessibility()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        # Overlay-Tick auf dem Main-Thread (50 ms)
        self._overlay_timer = rumps.Timer(overlay.tick, 0.05)
        self._overlay_timer.start()
        # Call-Detection alle 15 Sekunden
        self._call_timer = rumps.Timer(self._check_for_calls, 15)
        self._call_timer.start()

    def _check_accessibility(self):
        """Prüft ob Accessibility-Permission gesetzt ist — ohne sie funktioniert fn nicht."""
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to return true'],
            capture_output=True
        )
        if result.returncode != 0:
            subprocess.Popen([
                "osascript", "-e",
                'display notification "fn-Taste wird nicht erkannt. Bitte Accessibility-Zugriff in Systemeinstellungen → Datenschutz → Bedienungshilfen für Terminal/Python erlauben." with title "whispr ⚠️ Berechtigung fehlt"'
            ])

    def _setup_menu(self):
        """Menü einmalig aufbauen — Items werden später nur per .title aktualisiert."""
        self._item_dashboard     = rumps.MenuItem("Dashboard öffnen",      callback=self.open_dashboard)
        self._item_paste         = rumps.MenuItem("Letzten Text einfügen", callback=self.paste_last)
        self._item_lang          = rumps.MenuItem("",                      callback=self.toggle_language)
        self._item_mode          = rumps.MenuItem("",                      callback=self.toggle_mode_cb)
        self._item_meeting_start  = rumps.MenuItem("",                         callback=self.start_meeting_cb)
        self._item_meeting_stop   = rumps.MenuItem("",                         callback=self.stop_meeting_cb)
        self._item_meeting_manual = rumps.MenuItem("🎙 Meeting aufnehmen",    callback=self.start_manual_meeting_cb)
        self._item_test_start     = rumps.MenuItem("🧪 Test starten",         callback=self.start_test_cb)
        self._item_test_stop      = rumps.MenuItem("",                         callback=self.stop_test_cb)
        self._item_quit           = rumps.MenuItem("Beenden",                  callback=self.quit_app)
        self._item_meeting_start.set_callback(self.start_meeting_cb)
        self._item_meeting_stop.set_callback(self.stop_meeting_cb)
        self._item_test_stop.set_callback(self.stop_test_cb)
        self.menu = [
            self._item_dashboard,
            self._item_paste,
            rumps.separator,
            self._item_test_start,
            self._item_test_stop,
            self._item_meeting_manual,
            self._item_meeting_start,
            self._item_meeting_stop,
            rumps.separator,
            self._item_lang,
            self._item_mode,
            rumps.separator,
            self._item_quit,
        ]
        self._item_meeting_start.title = ""
        self._item_meeting_stop.title  = ""
        self._item_test_stop.title     = ""
        self._refresh_labels()
        self._refresh_test_menu()

    def _refresh_labels(self):
        """Nur die Labels aktualisieren — kein Menü-Rebuild."""
        lang_icons = {"de": "🇩🇪 Deutsch", "en": "🇬🇧 English"}
        self._item_lang.title = f"Sprache: {lang_icons.get(LANGUAGE, LANGUAGE)} (wechseln)"
        self._item_mode.title = f"Modus: {'🔁 Toggle' if toggle_mode else '⏺ Halten'} (wechseln)"

    def _refresh_test_menu(self):
        """Test starten/stoppen Items wechseln sich ab."""
        if test_active:
            self._item_test_start.title = ""
            self._item_test_stop.title  = "🛑 Test beenden"
        else:
            self._item_test_start.title = "🧪 Test starten"
            self._item_test_stop.title  = ""

    def open_dashboard(self, _):
        subprocess.Popen(["open", f"http://localhost:{PORT}"])

    def paste_last(self, _):
        if last_text:
            subprocess.run(["pbcopy"], input=last_text.encode("utf-8"))
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down'
            ])

    def toggle_language(self, _):
        global LANGUAGE
        cycle = {"de": "en", "en": "de"}
        LANGUAGE = cycle.get(LANGUAGE, "de")
        set_setting("language", LANGUAGE)
        self._refresh_labels()

    def toggle_mode_cb(self, _):
        global toggle_mode
        toggle_mode = not toggle_mode
        self._refresh_labels()

    def _check_for_calls(self, _=None):
        """Alle 15s: Prüfe ob ein Call läuft und zeige Menü-Item."""
        app, ctype = get_active_call_app()
        if app and not meeting_active and self._call_detected_app != app:
            self._call_detected_app  = app
            self._call_detected_type = ctype
            icon  = "👥" if ctype == "internal" else "📞"
            label = "Team-Call" if ctype == "internal" else "Externer Call"
            self._item_meeting_start.title = f"{icon} {label} erkannt — Aufzeichnen starten"
            self._item_meeting_stop.title  = ""
            subprocess.Popen(["osascript", "-e",
                f'display notification "Klick im Menü um Aufnahme zu starten" '
                f'with title "whispr {icon} {label} erkannt ({app})"'])
        elif not app and not meeting_active:
            self._call_detected_app        = None
            self._item_meeting_start.title = ""
            self._item_meeting_stop.title  = ""

    def start_manual_meeting_cb(self, _):
        """Manuell eine Meeting-Aufnahme starten — ohne Auto-Detection."""
        if meeting_active:
            return
        self._call_detected_app  = "Manuell"
        self._call_detected_type = "internal"
        start_meeting_recording("Manuell", "internal")
        self._item_meeting_manual.title = ""
        self._item_meeting_stop.title   = "⏹ Meeting stoppen & zusammenfassen"
        subprocess.Popen(["osascript", "-e",
            'display notification "Aufnahme l\u00e4uft \u2014 stoppe sie \u00fcber das Men\u00fc" '
            'with title "whispr \U0001f534 Meeting l\u00e4uft"'])

    def start_meeting_cb(self, _):
        if not self._call_detected_app:
            return
        start_meeting_recording(self._call_detected_app, self._call_detected_type)
        label = "Team-Call" if self._call_detected_type == "internal" else "Externer Call"
        self._item_meeting_start.title = ""
        self._item_meeting_stop.title  = f"⏹ {label} stoppen & zusammenfassen"
        subprocess.Popen(["osascript", "-e",
            'display notification "Aufnahme läuft — stoppe sie über das Menü" '
            'with title "whispr \U0001f534 Meeting läuft"'])

    def stop_meeting_cb(self, _):
        self._item_meeting_start.title  = ""
        self._item_meeting_stop.title   = ""
        self._item_meeting_manual.title = "🎙 Meeting aufnehmen"
        self._call_detected_app         = None
        threading.Thread(target=stop_and_process_meeting, daemon=True).start()

    def start_test_cb(self, _):
        # Session-Name per Dialog abfragen
        response = subprocess.run(
            ["osascript", "-e",
             'set n to text returned of (display dialog "Test-Session Name:" '
             'default answer "GmbH Flow v1" buttons {"Abbrechen", "Starten"} '
             'default button "Starten" with title "whispr 🧪 Testing")'],
            capture_output=True, text=True
        )
        name = response.stdout.strip()
        if not name or response.returncode != 0:
            return
        self._test_running = True
        start_test_session(name)
        if test_active:   # start_test_session setzt test_active nur wenn Permission OK
            self.title = "🧪"
            self._refresh_test_menu()
            subprocess.Popen(["osascript", "-e",
                f'display notification "Screenshots & URLs werden aufgezeichnet. '
                f'fn-Taste = Sprachnotiz anheften." '
                f'with title "whispr 🧪 Testing läuft: {name}"'])
        else:
            self._test_running = False

    def stop_test_cb(self, _):
        self._test_running = False
        self.title = "🎙"
        stop_test_session()
        self._refresh_test_menu()

    def quit_app(self, _):
        if stream is not None:
            stream.stop()
            stream.close()
        if meeting_stream is not None:
            meeting_stream.stop()
            meeting_stream.close()
        rumps.quit_application()

    def on_press(self, key):
        global fn_pressed, recording
        # Sicherer Vergleich — kein Crash wenn FN_KEY nicht existiert
        try:
            is_fn = (FN_KEY is not None and key == FN_KEY)
        except Exception:
            return
        if not is_fn:
            return

        if toggle_mode:
            if not recording:
                app_name   = get_frontmost_app()
                fn_pressed = True
                self.title = "🔴"
                overlay.show(app_name, LANGUAGE)
                if test_active:
                    threading.Thread(target=_capture_note_anchor_screenshot, daemon=True).start()
                try:
                    start_recording()
                except Exception as e:
                    print(f"[whispr] Mikrofon-Fehler: {e}")
                    overlay.hide()
                    fn_pressed = False
                    self.title = "🎙"
                    subprocess.Popen(["osascript", "-e",
                        'display notification "Mikrofon nicht verfügbar – bitte prüfen" with title "whispr ⚠️"'])
            else:
                fn_pressed = False
                self.title = "⏳"
                overlay.set_transcribing(LANGUAGE)
                threading.Thread(target=stop_and_transcribe, args=(self,), daemon=True).start()
        else:
            if not fn_pressed:
                app_name   = get_frontmost_app()
                fn_pressed = True
                self.title = "🔴"
                overlay.show(app_name, LANGUAGE)
                if test_active:
                    threading.Thread(target=_capture_note_anchor_screenshot, daemon=True).start()
                try:
                    start_recording()
                except Exception as e:
                    print(f"[whispr] Mikrofon-Fehler: {e}")
                    overlay.hide()
                    fn_pressed = False
                    self.title = "🎙"
                    subprocess.Popen(["osascript", "-e",
                        'display notification "Mikrofon nicht verfügbar – bitte prüfen" with title "whispr ⚠️"'])

    def on_release(self, key):
        global fn_pressed
        try:
            is_fn = (FN_KEY is not None and key == FN_KEY)
        except Exception:
            return
        if is_fn and fn_pressed and not toggle_mode:
            fn_pressed = False
            self.title  = "⏳"
            overlay.set_transcribing(LANGUAGE)
            threading.Thread(target=stop_and_transcribe, args=(self,), daemon=True).start()

if __name__ == "__main__":
    WhisprApp().run()
