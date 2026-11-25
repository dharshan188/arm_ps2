#!/usr/bin/env bash
# Start Sorti on this laptop (laptop = Raspberry Pi stand-in), then open the browser.
#   ./run.sh            English     ./run.sh --lang ta     ./run.sh --camera 1
#   ./run.sh --no-speaker --host 0.0.0.0   Pi, open the page from another device
cd "$(dirname "$0")" || exit 1
open_browser=true
for arg in "$@"; do
  if [ "$arg" = "--no-speaker" ]; then
    open_browser=false
    break
  fi
done
if "$open_browser"; then
  ( for i in $(seq 40); do sleep 1; curl -s localhost:8000/api/state >/dev/null && { xdg-open http://127.0.0.1:8000 >/dev/null 2>&1; break; }; done ) &
fi
exec env -u PYTHONPATH PATH="$HOME/ps2_et/bin:$PATH" "$HOME/ps2_et/bin/python" server.py "$@"
