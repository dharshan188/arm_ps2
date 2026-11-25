#!/usr/bin/env bash
PIDF="$HOME/sorti_vision.pid"
if [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null; then echo "stopped"; else echo "not running"; fi
rm -f "$PIDF"
