#!/usr/bin/env bash
# Start the live classifier; view it in a browser at http://<pi-ip>:8080
#   bash start_vision.sh            foreground (Ctrl+C stops)
#   bash start_vision.sh --bg       background, keeps running after you close SSH (log: ~/sorti_vision.log)
#   CAM=2 bash start_vision.sh      force camera /dev/video2      PORT=8090 ... other port
set -uo pipefail
cd "$(dirname "$0")/../vision" || exit 1
PY=(env -u PYTHONPATH "$HOME/ps2_et/bin/python")
PORT="${PORT:-8080}"
PIDF="$HOME/sorti_vision.pid"

if [ -z "${CAM:-}" ]; then   # Pi 5 has many internal /dev/video nodes; take the first node of the USB camera
  CAM=$(v4l2-ctl --list-devices 2>/dev/null | awk '/[Uu][Ss][Bb]|[Ww]ebcam|[Cc]amera/{f=1;next} f&&/\/dev\/video/{gsub(/[^0-9]/,"",$1);print $1;exit}')
  CAM="${CAM:-0}"
fi
IP=$(hostname -I | awk '{print $1}')
echo "camera: /dev/video$CAM"
echo "open on your laptop:  http://$IP:$PORT"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDF")). Stop it first:  bash $(dirname "$0")/stop_vision.sh"; exit 1
fi
if [ "${1:-}" = "--bg" ]; then
  nohup "${PY[@]}" pi_stream.py --camera "$CAM" --port "$PORT" > "$HOME/sorti_vision.log" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  echo "running in background, pid $(cat "$PIDF").  log: tail -f ~/sorti_vision.log"
else
  exec "${PY[@]}" pi_stream.py --camera "$CAM" --port "$PORT"
fi
