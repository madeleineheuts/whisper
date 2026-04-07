# Whispr — Voice Dictation App

> Internal tool for RAKETENSTART team. Built with faster-whisper (local AI transcription).

---

## What this app does

Whispr is a Mac menu bar app that lets you dictate text by holding a hotkey (right Option key). It records your voice, transcribes it locally using OpenAI's Whisper model, and pastes the text wherever your cursor is. No internet required. All processing happens on your machine.

**Current features:**
- Hold right Option key → record → release → text gets pasted
- Toggle mode (press once to start, press again to stop)
- Language switch: DE / EN
- Dictation history stored locally (SQLite)
- Custom dictionary for corrections
- Minimal overlay shows recording status

---

## Current stack (local version)

| What | How |
|------|-----|
| UI | `rumps` (Mac menu bar) |
| Audio | `sounddevice` |
| Transcription | `faster-whisper` (local model, runs on device) |
| Hotkey | `pynput` |
| Local API | `flask` (port 5173, internal only) |
| Database | SQLite (`~/.whispr.db`) |
| Overlay | PyObjC / AppKit |

---

## How to run (local, Mac only)

```bash
# 1. Install dependencies
bash setup.sh

# 2. Run the app
python3 whispr.py

# 3. First launch: grant Accessibility access
# System Settings → Privacy & Security → Accessibility → enable Terminal or Python
```

---

## 🚀 What needs to be built: Web version

**Goal:** Turn this into a web app the whole RAKETENSTART team can use — no local install required. Anyone should be able to sign up, open it in the browser, record, and get a transcription.

### Architecture

```
Browser (Next.js)
  → records audio via MediaRecorder API
  → sends audio file to API route
  → API route calls OpenAI Whisper API
  → returns transcription
  → user copies text or it auto-copies to clipboard

Supabase
  → Auth (email/password signup + login)
  → Database (stores dictation history per user)
```

### Tech stack decision

| What | Use | Why |
|------|-----|-----|
| Frontend + API routes | Next.js | Simple, Vercel-native |
| Hosting | Vercel | Free tier, auto-deploys from GitHub |
| Auth | Supabase Auth | We already use Supabase |
| Database | Supabase (Postgres) | Replace SQLite, same schema |
| Transcription | OpenAI Whisper API | No GPU/server needed, ~$0.006/min |

### What to build

**1. Auth**
- Email + password signup/login via Supabase Auth
- Simple login page, no OAuth needed for now

**2. Core transcription flow**
- Browser records audio → sends to `/api/transcribe`
- API route receives audio, sends to OpenAI Whisper API (`whisper-1` model)
- Returns transcript text to frontend
- Auto-copy to clipboard on completion

**3. History**
- Supabase table: `dictations` (user_id, text, word_count, duration, created_at)
- Show last 20 dictations per user

**4. Settings**
- Language toggle: DE / EN / auto
- That's it for v1

### Supabase table schema

```sql
create table dictations (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users not null,
  text text not null,
  word_count int default 0,
  duration real default 0,
  created_at timestamptz default now()
);

-- Row Level Security: users can only see their own dictations
alter table dictations enable row level security;
create policy "Users see own dictations" on dictations
  for all using (auth.uid() = user_id);
```

### Environment variables needed

```
OPENAI_API_KEY=sk-...
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

### Deployment

1. Push to GitHub (private repo)
2. Connect repo to Vercel
3. Add env variables in Vercel dashboard
4. Done — auto-deploys on every push to `main`

---

## Known bugs in the local version

**Overlay gets stuck on "Transcribing..."**
- The `RecordingOverlay` uses an `NSPanel` that floats above all windows (including Chrome)
- `overlay.hide()` is called correctly in the `finally` block of `stop_and_transcribe()`
- But if `faster_whisper`'s `model.transcribe()` hangs (e.g. empty/corrupt audio), the rumps main-thread timer (`tick()`) stops receiving the hide signal → overlay stays visible indefinitely
- Quick fix: `pkill -f whispr.py` and restart
- **This bug disappears entirely in the web version** — OpenAI Whisper API has its own timeout, there's no local model that can hang, and there's no overlay panel at all

---

## Out of scope for v1

- "Paste into active app" (browser security blocks this — clipboard copy is the workaround)
- Custom dictionary (can add in v2)
- Team admin / user management
- Mobile

---

## Questions / contact

Madeleine Heuts — madeleine@raketenstart.de
