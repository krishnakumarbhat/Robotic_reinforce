#!/usr/bin/env bash
# AGENT INSTRUCTION: Remote GPU dispatcher. Usage: ./run_gpu.sh <script.py> [args...]
# GPU_BACKEND=local|kaggle|colab (env wins over .env). COLAB_ACCOUNT=pro|old|auto.
# Lifecycle guarantees (do not weaken):
#  - colab: session is ALWAYS stopped (trap on EXIT/ERR/INT/TERM), even if exec crashes.
#  - colab: unique session name per job -> parallel jobs never stop each other.
#  - kaggle: polls to terminal state (kernels auto-stop), then fetches outputs; warns if
#    outputs are large (checkpoints belong on the HF Hub, never in /kaggle/working).
#  - every job appends one line to usage.log (backend, minutes, rc) for quota accounting.
# Knobs: GPU_MAX_MIN (hard wall per job, default 240), COLAB_GPU_TYPE (T4|L4|A100),
#   COLAB_UPLOADS="local1:remote1,local2:remote2", COLAB_DOWNLOADS="remote1:local1,...".
set -euo pipefail
cd "$(dirname "$0")"
CLI_BACKEND="${GPU_BACKEND:-}"
CLI_COLAB_ACCT="${COLAB_ACCOUNT:-}"
CLI_KAGG_ACCEL="${KAGGLE_ACCELERATOR:-__UNSET__}"
set -a; source .env; set +a
set -a; source colab/.env 2>/dev/null || true; set +a

BACKEND="${CLI_BACKEND:-${GPU_BACKEND:-local}}"
COLAB_ACCT="${CLI_COLAB_ACCT:-${COLAB_ACCOUNT:-pro}}"
# Empty CLI value forces CPU kernel; unset CLI keeps .env.
[ "$CLI_KAGG_ACCEL" != "__UNSET__" ] && KAGGLE_ACCELERATOR="$CLI_KAGG_ACCEL"
SCRIPT="$(readlink -f "${1:?usage: ./run_gpu.sh script.py [args...]}")"
shift || true
KSLUG="${KAGGLE_KERNEL_SLUG:-}"
MAX_MIN="${GPU_MAX_MIN:-240}"
USE_COLAB="$(dirname "$(readlink -f "$0")")/use-colab.sh"
[ -f "$USE_COLAB" ] || USE_COLAB="/media/pope/projecteo/connect/connect_gpu/use-colab.sh"
T_START=$(date +%s)
usage_log() { echo "$(date -u +%FT%TZ) backend=$BACKEND script=$(basename "$SCRIPT") min=$(( ($(date +%s) - T_START) / 60 )) rc=$1" >> usage.log; }

colab_run() { # $1=acct, rest = script args. new -> uploads -> exec -> downloads -> stop (trap)
  local acct="$1"; shift
  # shellcheck disable=SC1090
  source "$USE_COLAB" "$acct" >&2
  local S="${COLAB_SESSION_NAME:-research}-$$"
  local C="$COLAB_CONFIG"
  trap 'colab --config "$C" stop -s "$S" >/dev/null 2>&1 || true; echo "[run_gpu] session $S stopped" >&2' EXIT INT TERM
  colab --config "$C" new -s "$S" --gpu "${COLAB_GPU_TYPE:-T4}" || { trap - EXIT INT TERM; return 97; }
  local pair
  IFS=',' read -ra UPS <<< "${COLAB_UPLOADS:-}"
  for pair in "${UPS[@]}"; do [ -n "$pair" ] && colab --config "$C" upload -s "$S" "${pair%%:*}" "${pair#*:}"; done
  local rc=0
  local LOGF="usage_${S}.out"
  colab --config "$C" exec -s "$S" -f "$SCRIPT" --timeout $(( MAX_MIN * 60 )) "$@" 2>&1 | tee "$LOGF" || rc=$?
  # colab exec exits 0 even when the cell raises -> detect the traceback ourselves
  grep -qE "Traceback \(most recent call last\)|SystemExit.*[1-9]" "$LOGF" && rc=${rc/#0/1}
  rm -f "$LOGF"
  IFS=',' read -ra DNS <<< "${COLAB_DOWNLOADS:-}"
  for pair in "${DNS[@]}"; do [ -n "$pair" ] && colab --config "$C" download -s "$S" "${pair%%:*}" "${pair#*:}" || true; done
  colab --config "$C" stop -s "$S" || true
  trap - EXIT INT TERM
  return $rc
}

rc=0
case "$BACKEND" in
  local)
    python3 "$SCRIPT" "$@" || rc=$?
    ;;
  kaggle)
    [ -n "$KSLUG" ] || { echo "KAGGLE_KERNEL_SLUG unset" >&2; exit 2; }
    cp "$SCRIPT" kaggle/code.py
    jq '.code_file="code.py"' kaggle/kernel-metadata.json > kaggle/kernel-metadata.tmp
    if [ -z "${KAGGLE_ACCELERATOR:-}" ]; then
      jq 'del(.machine_shape)' kaggle/kernel-metadata.tmp > kaggle/kernel-metadata.json
    else
      jq --arg m "$KAGGLE_ACCELERATOR" '.machine_shape=$m' kaggle/kernel-metadata.tmp > kaggle/kernel-metadata.json
    fi
    rm -f kaggle/kernel-metadata.tmp
    kaggle kernels push -p kaggle
    # Accelerator comes from kernel-metadata.json machine_shape ONLY (the installed
    # kaggle CLI has no --accelerator flag). Empty KAGGLE_ACCELERATOR above deletes
    # machine_shape above -> CPU kernel, zero GPU quota.
    deadline=$(( T_START + MAX_MIN * 60 + 900 ))   # +15 min queue slack
    st=""
    while [ "$(date +%s)" -lt "$deadline" ]; do
      st=$(kaggle kernels status "$KSLUG" 2>&1 | tail -1)
      case "$st" in *COMPLETE*|*ERROR*|*CANCEL*|*FAIL*) break ;; esac
      sleep 60
    done
    echo "[run_gpu] kaggle final: $st" >&2
    mkdir -p outputs
    kaggle kernels output "$KSLUG" -p outputs || rc=$?
    sz=$(du -sm outputs | cut -f1)
    [ "$sz" -gt 200 ] && echo "[run_gpu] WARNING outputs/ is ${sz}MB: push checkpoints to HF Hub, keep /kaggle/working small" >&2
    case "$st" in *COMPLETE*) ;; *) rc=${rc:-0}; [ "$rc" -eq 0 ] && rc=1 ;; esac
    ;;
  colab)
    if [ "$COLAB_ACCT" = "auto" ]; then
      # fall back ONLY when the Pro session could not be created (quota/auth), never
      # re-run a job that started and failed (that would double-burn units).
      colab_run pro "$@" || { rc=$?; if [ "$rc" -eq 97 ]; then echo "pro unavailable -> old" >&2; rc=0; colab_run old "$@" || rc=$?; fi; }
    else
      colab_run "$COLAB_ACCT" "$@" || rc=$?
    fi
    ;;
  *)
    echo "Unknown GPU_BACKEND: $BACKEND (local|kaggle|colab)" >&2; exit 1 ;;
esac
usage_log "$rc"
exit "$rc"
