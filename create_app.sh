#!/bin/bash
# whispr.app erstellen — einmal ausführen, danach im Launchpad verfügbar
set -e

WHISPR_DIR="$HOME/Downloads/Claude/Whispr"
APP_DIR="$HOME/Applications/whispr.app"

echo "🚀 Erstelle whispr.app..."

# --- 1. App-Bundle Struktur ---
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# --- 2. Launcher-Script ---
cat > "$APP_DIR/Contents/MacOS/whispr" << 'LAUNCHER'
#!/bin/bash
# PATH für Homebrew / pyenv / conda laden
source ~/.zprofile 2>/dev/null || true
source ~/.zshrc    2>/dev/null || true

DIR="$HOME/Downloads/Claude/Whispr"
cd "$DIR"

# Bereits gestartet? Nicht doppelt starten.
if pgrep -f "python3 $DIR/whispr.py" > /dev/null 2>&1; then
    osascript -e 'display notification "whispr läuft bereits." with title "whispr"'
    exit 0
fi

# .env laden falls vorhanden
if [ -f "$DIR/.env" ]; then
    set -a
    source "$DIR/.env"
    set +a
fi

exec python3 "$DIR/whispr.py"
LAUNCHER
chmod +x "$APP_DIR/Contents/MacOS/whispr"

# --- 3. Info.plist ---
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>whispr</string>
    <key>CFBundleDisplayName</key>
    <string>whispr</string>
    <key>CFBundleIdentifier</key>
    <string>de.raketenstart.whispr</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>whispr</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>whispr benötigt das Mikrofon für lokale Sprachdiktierung.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>whispr benötigt AppleScript zur automatischen Texteingabe.</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
</dict>
</plist>
PLIST

# --- 4. App-Icon erstellen ---
echo "🎨 Erstelle App-Icon..."

python3 << 'PYTHON'
import os, math
try:
    from PIL import Image, ImageDraw
except ImportError:
    os.system("pip3 install Pillow --break-system-packages -q")
    from PIL import Image, ImageDraw

def make_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Hintergrund: dunkles Violett mit abgerundeten Ecken
    pad = int(size * 0.07)
    radius = int(size * 0.22)
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=radius,
        fill=(38, 18, 72, 255)
    )

    # Dezenter Gradient-Effekt (helleres Oval oben)
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.ellipse([pad, pad, size - pad, int(size * 0.6)], fill=(80, 40, 140, 60))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Mikrofon-Körper
    cx = size / 2
    mw = size * 0.20
    mh = size * 0.30
    my = size * 0.18
    lw = max(2, int(size * 0.042))
    draw.rounded_rectangle(
        [cx - mw / 2, my, cx + mw / 2, my + mh],
        radius=mw / 2,
        fill=(255, 255, 255, 255)
    )

    # Mikrofon-Bogen
    arc_r = size * 0.26
    arc_cy = my + mh * 0.62
    draw.arc(
        [cx - arc_r, arc_cy - arc_r * 0.8, cx + arc_r, arc_cy + arc_r * 0.8],
        start=0, end=180,
        fill=(255, 255, 255, 230),
        width=lw
    )

    # Stiel
    stiel_top = arc_cy + arc_r * 0.75
    stiel_bot = size * 0.83
    draw.line([cx, stiel_top, cx, stiel_bot], fill=(255, 255, 255, 230), width=lw)
    base_hw = size * 0.16
    draw.line([cx - base_hw, stiel_bot, cx + base_hw, stiel_bot],
              fill=(255, 255, 255, 230), width=lw)

    # Kleiner Raketenstart-Akzent: 3 kleine Punkte unten rechts
    dot_r = max(2, int(size * 0.025))
    dot_y = int(size * 0.84)
    for i, col in enumerate([(255,100,100,200),(255,180,60,200),(100,220,130,200)]):
        dx = int(cx + size * 0.22 + i * (dot_r * 2.8))
        if dx + dot_r < size - pad:
            draw.ellipse([dx - dot_r, dot_y - dot_r, dx + dot_r, dot_y + dot_r], fill=col)

    return img

# Iconset
iconset = os.path.expanduser("~/Downloads/Claude/Whispr/AppIcon.iconset")
os.makedirs(iconset, exist_ok=True)

for s in [16, 32, 64, 128, 256, 512, 1024]:
    make_icon(s).save(f"{iconset}/icon_{s}x{s}.png")
    if s <= 512:
        make_icon(s * 2).save(f"{iconset}/icon_{s}x{s}@2x.png")

print("✅ Icon-PNGs erstellt")
PYTHON

# iconset → .icns
ICONSET="$WHISPR_DIR/AppIcon.iconset"
ICNS="$APP_DIR/Contents/Resources/AppIcon.icns"
iconutil -c icns "$ICONSET" -o "$ICNS"
rm -rf "$ICONSET"
echo "✅ AppIcon.icns erstellt"

# --- 5. Launchpad-Cache neu laden ---
touch "$APP_DIR"
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister \
    -f "$APP_DIR" 2>/dev/null || true

echo ""
echo "✅ whispr.app ist fertig!"
echo ""
echo "📍 Gespeichert unter: ~/Applications/whispr.app"
echo "🔍 In Spotlight: Command+Leertaste → 'whispr' eingeben"
echo "🚀 Im Launchpad: erscheint nach nächstem Login automatisch"
echo ""
echo "💡 Tipp: Ziehe whispr.app in das Dock für Schnellzugriff."
