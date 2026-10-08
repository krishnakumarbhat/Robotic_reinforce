#!/bin/bash
# kaggle-watch.sh: one-shot blitz monitor. Prints kernel status, GPU-hour burn,
# reserve, loop state, disk. No args.
cd "$(dirname "$0")" || exit 1
echo "== kaggle =="; timeout 60 kaggle kernels status krishnakumarbhat/remote-train 2>&1 | tail -1
echo "== loop =="; tail -1 .autoresearch-loop.log 2>&1 | cut -c1-110
echo "== local probes =="; ps aux 2>&1 | grep "[i]6b_local\|[l]ocal_eval" | awk '{print $11, $12}' | head -3
echo "== disk =="; df -h / 2>&1 | tail -1 | awk '{print $3" used / "$4" free ("$5")"}'
echo "== quota (manual: read sidebar, update RESERVE) =="
echo "spent-today-est: see worklog; RESERVE rule: stop all pushes at 1.0h remaining"
