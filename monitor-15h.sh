#!/bin/bash
# monitor-15h.sh — hourly watchdog for the 15h breakthrough run.
# Logs health to MONITOR-15H.md; STOPS the loop if / drops below 5GB (crash protection).
# Usage: nohup ./monitor-15h.sh >/dev/null 2>&1 &
cd "$(dirname "$0")"
LOG="MONITOR-15H.md"
DISK_MIN_GB=5
log() { echo "[$1] $2" >> "$LOG"; }
while true; do
    ts=$(date '+%F %T')
    avail_gb=$(($(df / | tail -1 | awk '{print $4}')/1024/1024))
    driver="DEAD"; kill -0 "$(cat .autoresearch-loop.lock 2>/dev/null)" 2>/dev/null && driver="ALIVE"
    workers=$(pgrep -f "[o]pencode run --auto" 2>/dev/null | wc -l)
    runs=$(wc -l < autoresearch_research.jsonl 2>/dev/null || echo 0)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
    if (( avail_gb < DISK_MIN_GB )); then
        ~/.config/opencode/scripts/autoresearch-loop.sh stop >> "$LOG" 2>&1 || true
        log "$ts" "DISK-GUARD TRIPPED: ${avail_gb}G < ${DISK_MIN_GB}G — loop STOPPED to protect system"
        exit 0
    fi
    log "$ts" "driver=$driver workers=$workers jsonl=$runs disk=${avail_gb}G gpu=[$gpu]"
    sleep 3600
done
