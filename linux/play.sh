#!/usr/bin/env bash
# Linux counterpart of scripts/play.ps1 (NVIDIA preset), on GE-Proton11-14's Wine.
#   GAME=/path/to/title linux/play.sh [series-s|series-x] [victory args...]
# GAME is the folder with game_scarlett_master.exe. Env: PROTON (GE-Proton folder),
# VICTORY_PREFIX, VICTORY_HOME (prefixes, logs, vkd3d cache; default ~/.local/share/victory),
# BUILD (diag|release) or EXE, WINDOWED=1, NOPANEL=1, SAFE=1 (play.ps1's -Safe).
# The prefix needs Microsoft's d3dcompiler_47 (Wine's builtin rejects the runtime HLSL):
#   WINEPREFIX=<prefix>/pfx WINE=$PROTON/files/bin/wine winetricks -q d3dcompiler_47
# Runs with no network (unshare -rn, loopback only). Set NET=1 to allow it.
set -euo pipefail
cd "$(dirname "$0")/.."
CONSOLE=${1:-series-x}; shift || true
[ -n "${GAME:-}" ] && [ -f "$GAME/game_scarlett_master.exe" ] || { echo "set GAME to the folder with game_scarlett_master.exe"; exit 1; }
GAME=$(realpath "$GAME")
VH=${VICTORY_HOME:-$HOME/.local/share/victory}
# GE-Proton11-14 with linux/wineserver's faster server when built, else the plain one.
VT=$HOME/.steam/root/compatibilitytools.d/GE-Proton11-14-viewtree
[ -x "$VT/files/bin/wineserver.ge" ] && DEF=$VT || DEF=/usr/share/steam/compatibilitytools.d/proton-ge-custom
PROTON=${PROTON:-$DEF}
EXE=${EXE:-target/x86_64-pc-windows-msvc/${BUILD:-diag}/victory.exe}
[ -f "$EXE" ] || { echo "build $EXE first (see README)"; exit 1; }

# ponytail: same values as play.ps1's base + NVIDIA + non-safe settings; keep in sync by hand.
export VICTORY_TEXTURE_NO_WAIT=1 VICTORY_RELAY=2 VICTORY_WIDE_BACK=1 VICTORY_CARD_REHASH_RUNS=1 \
  VICTORY_JOIN_COPIES=1 VICTORY_INDIRECT_WINDOW=1 VICTORY_RELAY_COPY=1 VICTORY_CARD_STEPS=2 \
  VICTORY_TITLE_PRIORITIES=0 VICTORY_SHOW_AFTER=1 VICTORY_AMBIENT=0.8 VICTORY_D3D12_DRED=light \
  VICTORY_THIN_THREADS=12 VICTORY_UNDER_THREADS=8 VICTORY_LAND_THREADS=6
# SAFE=1 is play.ps1's -Safe: none of the settings still on trial.
if [ "${SAFE:-0}" = 1 ]; then
  export VICTORY_UNDER_WIDE=0 VICTORY_LAND_ECHO=0
else
  export VICTORY_MAP_THREADS=auto VICTORY_PLACED_ORDER=1 \
    VICTORY_THIN=1 VICTORY_JOIN_PARTS=8 VICTORY_UPLOAD_AHEAD=1 VICTORY_INDIRECT_BATCH=2
fi
[ "${WINDOWED:-0}" = 1 ] || export VICTORY_FULLSCREEN=1
[ "${NOPANEL:-0}" = 1 ] || export VICTORY_PANEL=1

PREFIX=${VICTORY_PREFIX:-$VH/prefix-$(awk "{print \$2}" "$PROTON/version")}
# ponytail: `proton run` exits 1 with no output for victory.exe, so Proton only builds the
# prefix (vkd3d-proton, DXVK) and the title runs on its wine with Proton's DLL overrides.
if [ ! -d "$PREFIX/pfx" ]; then
  mkdir -p "$PREFIX"
  STEAM_COMPAT_DATA_PATH=$PREFIX STEAM_COMPAT_CLIENT_INSTALL_PATH=$HOME/.local/share/Steam \
    unshare -rn "$PROTON/proton" run cmd /c exit || true
fi
export VKD3D_SHADER_CACHE_PATH=${VKD3D_SHADER_CACHE_PATH:-$VH/vkd3d}; mkdir -p "$VKD3D_SHADER_CACHE_PATH"
export WINEPREFIX=$PREFIX/pfx WINEDEBUG=${WINEDEBUG:--all} \
  WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11,d3d10core,d3d9=n;${WINEDLLOVERRIDES:-}"
run() {
    if [ "${NET:-0}" = 1 ]; then
        "$PROTON/files/bin/wine" "$EXE" "$@"
    else
        unshare -rn sh -c 'ip link set lo up 2>/dev/null; exec "$@"' sh "$PROTON/files/bin/wine" "$EXE" "$@"
    fi
}

# The executable checks pretranslation against its shader cache key.
run shaders --if-needed || echo "shader pretranslation failed; continuing"
# Every run's output is kept in $VH/last-run.log (previous one: .1).
LOG=$VH/last-run.log
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.1"
echo "log: $LOG"
# Collect both statuses before errexit can skip the exit report.
set +e
run run "Z:${GAME//\//\\}\\game_scarlett_master.exe" --console "$CONSOLE" --execute "$@" 2>&1 | tee "$LOG"
status=("${PIPESTATUS[@]}")
set -e
rc=${status[0]}
printf 'exit code: %s\n' "$rc" | tee -a "$LOG" || status[1]=1
if (( rc != 0 )); then exit "$rc"; fi
exit "${status[1]}"
