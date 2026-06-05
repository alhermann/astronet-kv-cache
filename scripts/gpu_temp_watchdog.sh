#!/bin/bash
# Temperature watchdog: kills the hottest training process if any
# Titan RTX exceeds 83 C for more than 60 s.
#
# Monitors cuda:0 and cuda:1 (the two Titan RTX cards; GT 1030 is
# ignored). Logs to logs/training/gpu_temp_watchdog.log.

set -uo pipefail
LOG=/home/alexander/Schreibtisch/AstroNet/logs/training/gpu_temp_watchdog.log
THRESH=89
SUSTAIN_SEC=120
CHECK_INTERVAL=10

mkdir -p "$(dirname "$LOG")"
echo "[watchdog] start $(date)  thresh=${THRESH}C sustain=${SUSTAIN_SEC}s" >> "$LOG"

declare -A hot_since
hot_since[1]=0  # Titan RTX at nvidia-smi index 1
hot_since[2]=0  # Titan RTX at nvidia-smi index 2

while true; do
    # Query temperatures for the two Titans (nvidia-smi index 1 and 2)
    readarray -t TEMPS < <(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | awk -F', ' '$1==1 || $1==2 {print $0}')
    now=$(date +%s)
    for line in "${TEMPS[@]}"; do
        idx=${line%%, *}
        t=${line##*, }
        if [ "$t" -ge "$THRESH" ]; then
            if [ "${hot_since[$idx]}" -eq 0 ]; then
                hot_since[$idx]=$now
                echo "[watchdog] $(date): GPU $idx hot start (${t}C)" >> "$LOG"
            else
                hot_dur=$((now - hot_since[$idx]))
                if [ "$hot_dur" -ge "$SUSTAIN_SEC" ]; then
                    # Kill the train_hybrid process pinned to that PyTorch device
                    # nvidia-smi index 1 = cuda:0, index 2 = cuda:1
                    case "$idx" in
                        1) pyt_dev="cuda:0" ;;
                        2) pyt_dev="cuda:1" ;;
                    esac
                    victim=$(pgrep -af "train_hybrid.py.*--device $pyt_dev" | awk '{print $1}' | head -1)
                    if [ -n "$victim" ]; then
                        echo "[watchdog] $(date): KILLING pid $victim on $pyt_dev (GPU $idx at ${t}C for ${hot_dur}s)" >> "$LOG"
                        kill -TERM "$victim"
                        sleep 5
                        kill -KILL "$victim" 2>/dev/null
                    fi
                    hot_since[$idx]=0
                fi
            fi
        else
            if [ "${hot_since[$idx]}" -ne 0 ]; then
                echo "[watchdog] $(date): GPU $idx cooled to ${t}C" >> "$LOG"
            fi
            hot_since[$idx]=0
        fi
    done
    sleep $CHECK_INTERVAL
done
