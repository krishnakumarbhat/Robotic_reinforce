#!/usr/bin/env bash
# Resilient launcher for the FROZEN autoresearch-loop driver (do not add logic to the driver itself).
# 1. Pre-flight probes a worker seat; never launches into a dead provider pool.
# 2. Pins relay to verified seats via launch env (loader: launch wins over file).
# 3. If the driver quits on consecutive-stale (transient outage), probe + relaunch
#    with backoff, bounded rounds — rides out 500s instead of dying at 17:25-style events.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /tmp/model-probe  # probes + driver relay use --dir here; a wiped /tmp made ALL seats look dead

# REAPER (2026-09-30): every dead driver/loop/reboot leaks its `opencode` workers, which
# then burn CPU/RAM for days (4 orphans held ~3GB + swap). Reap before each driver start.
# SAFETY (do not weaken): never kill (1) any pid in our own ancestry chain (the launcher
# itself often runs UNDER an opencode TUI session!), (2) children of a live driver loop,
# (3) PIDs from /tmp/.autoresearch_global.lock.pid while that pid is alive. Only exact
# `opencode .` TUIs older than 2h, and `opencode run` workers older than 40 min
# (watchdog is 20 min, so anything older is over budget = orphaned or stuck).
reap_orphans() {
  local -A keep=()
  local p=$BASHPID
  while [ "$p" != "1" ] && [ -n "$p" ]; do keep[$p]=1; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); done
  local lockpid
  lockpid=$(cat /tmp/.autoresearch_global.lock.pid 2>/dev/null || echo "")
  [ -n "$lockpid" ] && kill -0 "$lockpid" 2>/dev/null && keep[$lockpid]=1
  for p in $(pgrep -f "^opencode \.$" 2>/dev/null); do
    [ -n "${keep[$p]:-}" ] && continue
    local age
    age=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$age" ] && [ "$age" -gt 7200 ] && { kill "$p" 2>/dev/null && echo "[reap] stale TUI $p"; }
  done
  for p in $(pgrep -f "opencode run --auto" 2>/dev/null); do
    [ -n "${keep[$p]:-}" ] && continue
    if [ -n "$lockpid" ] && [ -f "/proc/$lockpid/cmdline" ]; then :; else
      local age
      age=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
      [ -n "$age" ] && [ "$age" -gt 2400 ] && { kill "$p" 2>/dev/null && echo "[reap] orphan worker $p"; }
    fi
  done
}
PROBE='reply with exactly: probe-ok'
SEAT_A="openrouter/thinkingmachines/inkling:free"
SEAT_B="opencode-responses/muse-spark-1.3-contributor-free"
probe() { timeout 100 opencode run --auto --dir /tmp/model-probe -m "$1" "$PROBE" 2>&1 | grep -q probe-ok; }
ROUNDS=0
SEAT_C="opencode/mimo-v2.6-flash-free"
SEAT_D="opencode/space-bunny-free"  # unlimited while alive; fast-fail when throttled
while (( ROUNDS < 48 )); do  # ponytail: rides out ~8h free-pool outages
  # A driver already holds the machine-wide lock (ours or another project's): wait, don't
  # burn rounds spawning drivers that exit rc=1 on the lock.
  if ! flock -n /tmp/.autoresearch_global.lock true 2>/dev/null; then
    echo "[launch] global lock held (pid $(cat /tmp/.autoresearch_global.lock.pid 2>/dev/null)) — waiting 5min"
    sleep 300; continue
  fi
  if probe "$SEAT_A" || probe "$SEAT_B" || probe "$SEAT_C" || probe "$SEAT_D"; then
    reap_orphans
    echo "[launch] seat alive — starting driver (round $ROUNDS)"
    AUTORESEARCH_BRAIN_LIST="$SEAT_B" \
    AUTORESEARCH_DIRECTIVE=".autoresearch-directive-aegis.md" \
      nohup ~/.config/opencode/scripts/autoresearch-loop.sh autoresearch_research >/dev/null 2>&1 &
    DRIVER=$!
    wait "$DRIVER"; rc=$?
    echo "[launch] driver exited rc=$rc"
    grep -q '"status":"complete"' autoresearch_research.jsonl 2>/dev/null && { echo "[launch] COMPLETE"; exit 0; }
  else
    echo "[launch] all probe seats dead — sleeping 10min"
    sleep 600
  fi
  ROUNDS=$((ROUNDS + 1))
  sleep 30
done
echo "[launch] gave up after $ROUNDS rounds"; exit 1
