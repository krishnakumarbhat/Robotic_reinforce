#!/usr/bin/env bash
# One-shot launcher: vcan0 setup -> CANalyst-II <-> vcan0 bridge -> lerobot-teleoperate.
#
# Usage:  ./teleop.sh          (asks for the sudo password once, up front)
#         sudo ./teleop.sh     (no prompts; uv commands are run as the invoking user)
#
# For zero prompts without sudo, add a sudoers drop-in (sudo visudo -f /etc/sudoers.d/vcan):
#   leevai ALL=(root) NOPASSWD: /usr/sbin/modprobe vcan, /usr/sbin/ip link add dev vcan0 type vcan, /usr/sbin/ip link set up vcan0
#
# Env overrides: LEROBOT_DIR, TELEOP_PORT, BRIDGE_TIMEOUT (seconds).

set -uo pipefail

BRIDGE_MSG="The bridge could not be started or died suddenly, ensure the follower arm is powered, connected and retry after restarting the motor bus"
TELEOP_MSG="Teleoperation failed, verify both arms are powered, connected and are in relatively similar initial poses. Bridge running... ( logs accessible at 'candump vcan0')"

if [[ -t 2 ]]; then RED=$'\e[1;31m'; YEL=$'\e[1;33m'; GRN=$'\e[1;32m'; RST=$'\e[0m'; else RED= YEL= GRN= RST=; fi
info() { echo "${GRN}[teleop.sh]${RST} $*" >&2; }
warn() { echo "${YEL}[teleop.sh] $*${RST}" >&2; }
err()  { echo "${RED}[teleop.sh] $*${RST}" >&2; }

# --- privileges -------------------------------------------------------------
# Only the vcan setup needs root. uv lives in the user's ~/.local/bin and must not
# touch the .venvs as root, so when started via sudo we drop back to $SUDO_USER.
if [[ $EUID -eq 0 ]]; then
    SUDO=()
    if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != root ]]; then
        USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        AS_USER=(sudo -u "$SUDO_USER" -H env "PATH=$USER_HOME/.local/bin:$PATH")
    else
        USER_HOME=$HOME
        AS_USER=()
        warn "Running as plain root; uv commands will run as root."
    fi
else
    SUDO=(sudo)
    USER_HOME=$HOME
    AS_USER=()
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(dirname "$SCRIPT_DIR")
LEROBOT_DIR=${LEROBOT_DIR:-$USER_HOME/dev/lerobot-configured}
TELEOP_PORT=${TELEOP_PORT:-/dev/ttyACM0}
BRIDGE_TIMEOUT=${BRIDGE_TIMEOUT:-30}
BRIDGE_LOG=$(mktemp -t teleop-bridge.XXXXXX.log)

BRIDGE_PID=
TELEOP_PID=
INTERRUPTED=

alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }
any_alive() { local p; for p; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
# Wait up to $1 seconds for all remaining pids to exit.
wait_gone() { local n=$(($1 * 4)); shift; while (( n-- > 0 )); do any_alive "$@" || return 0; sleep 0.25; done; ! any_alive "$@"; }
# A pid plus all its descendants (sudo -> uv -> python), deepest first.
tree() { local c; for c in $(pgrep -P "$1"); do tree "$c"; done; echo "$1"; }

# Escalate SIGINT -> SIGTERM -> SIGKILL on a whole process tree. $1 = seconds of
# grace before our own SIGINT (used when the terminal already delivered Ctrl+C).
stop_tree() {
    local pre=$1 pids; shift
    pids=("$@")
    any_alive "${pids[@]}" || return 0
    (( pre > 0 )) && wait_gone "$pre" "${pids[@]}" && return 0
    kill -INT "${pids[@]}" 2>/dev/null;  wait_gone 5 "${pids[@]}" && return 0
    kill -TERM "${pids[@]}" 2>/dev/null; wait_gone 3 "${pids[@]}" && return 0
    kill -KILL "${pids[@]}" 2>/dev/null
}

cleanup() {
    # Ignore further Ctrl+C / TERM so a second press can't orphan the bridge mid-cleanup.
    trap '' INT TERM HUP
    trap - EXIT
    if alive "$TELEOP_PID"; then
        info "Stopping teleoperation..."
        # On Ctrl+C teleop already got SIGINT from the terminal: give its disconnect
        # time to finish rather than interrupting it with a second SIGINT.
        stop_tree $([[ -n "$INTERRUPTED" ]] && echo 10 || echo 0) $(tree "$TELEOP_PID")
    fi
    if alive "$BRIDGE_PID"; then
        info "Stopping bridge..."
        # setsid made the bridge's PID its process-group id: signal the whole group.
        stop_tree 0 $(tree "$BRIDGE_PID")
        kill -KILL -- "-$BRIDGE_PID" 2>/dev/null
    fi
    rm -f "$BRIDGE_LOG"
    info "Stopped."
}
trap cleanup EXIT
trap 'INTERRUPTED=1; echo >&2; info "Interrupted, shutting down..."; exit 130' INT
trap 'info "Terminated, shutting down..."; exit 143' TERM
trap 'exit 129' HUP   # terminal window closed
# Output piped somewhere (e.g. `| tee run.log`) whose reader died on Ctrl+C must not
# kill us via SIGPIPE before cleanup runs; the failed write is simply dropped.
trap '' PIPE

bridge_failed() { err "$BRIDGE_MSG"; exit 1; }

# --- step 1: vcan0 + bridge -------------------------------------------------
info "Setting up vcan0..."
"${SUDO[@]}" modprobe vcan || { err "modprobe vcan failed"; exit 1; }
if ! ip link show vcan0 &>/dev/null; then
    "${SUDO[@]}" ip link add dev vcan0 type vcan || { err "Could not create vcan0"; exit 1; }
fi
"${SUDO[@]}" ip link set up vcan0 || { err "Could not bring up vcan0"; exit 1; }

info "Starting bridge (log: $BRIDGE_LOG)..."
# setsid puts the whole uv -> python tree in its own process group so it can be
# torn down as a unit. PYTHONUNBUFFERED makes the "Bridge running." line show up.
# Bash starts `&` jobs with SIGINT ignored and Python keeps that, so it would never
# see KeyboardInterrupt; env --default-signal undoes it (and our SIGPIPE ignore).
# The log pipe ignores Ctrl+C so the bridge's shutdown output still gets through.
cd "$BRIDGE_DIR" || exit 1
setsid env --default-signal=INT,PIPE "${AS_USER[@]}" env PYTHONUNBUFFERED=1 uv run socketcan_virtual_converter.py \
    < /dev/null > >(trap '' INT; tee "$BRIDGE_LOG" | sed -u 's/^/[bridge] /') 2>&1 &
BRIDGE_PID=$!

deadline=$((SECONDS + BRIDGE_TIMEOUT))
until grep -q "Bridge running." "$BRIDGE_LOG" 2>/dev/null; do
    alive "$BRIDGE_PID" || bridge_failed
    (( SECONDS < deadline )) || { err "Bridge did not report ready within ${BRIDGE_TIMEOUT}s."; bridge_failed; }
    sleep 0.5
done
sleep 1   # settle: make sure it didn't die right after starting
alive "$BRIDGE_PID" || bridge_failed
info "Bridge is up (vcan0 <-> CANalyst-II)."

# --- step 2: teleoperation --------------------------------------------------
# Stays in the terminal's foreground process group with a real stdin, so Ctrl+C
# reaches it directly and any calibration prompts still work.
info "Starting teleoperation..."
cd "$LEROBOT_DIR" || { err "LeRobot dir not found: $LEROBOT_DIR"; exit 1; }
TTY_IN=/dev/stdin
{ : < /dev/tty; } 2>/dev/null && TTY_IN=/dev/tty
env --default-signal=INT,PIPE "${AS_USER[@]}" uv run lerobot-teleoperate \
    --robot.type=damiao_follower --robot.port=vcan0 --robot.id=damiao_follower_01 \
    --teleop.type=sts_b601_leader --teleop.port="$TELEOP_PORT" --teleop.id=sts_b601_leader_01 \
    --fps=20 < "$TTY_IN" &
TELEOP_PID=$!

# --- supervise ---------------------------------------------------------------
while true; do
    if ! alive "$BRIDGE_PID"; then
        wait "$BRIDGE_PID" 2>/dev/null
        bridge_failed
    fi
    if [[ -n "$TELEOP_PID" ]] && ! alive "$TELEOP_PID"; then
        wait "$TELEOP_PID"; rc=$?
        TELEOP_PID=
        if (( rc != 0 )); then
            err "$TELEOP_MSG"
        else
            info "Teleoperation exited. Bridge still running; press Ctrl+C to stop."
        fi
    fi
    sleep 0.5
done
