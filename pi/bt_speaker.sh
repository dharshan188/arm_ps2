#!/usr/bin/env bash
# Put the paired Bluetooth speaker in A2DP (hi-fi stereo, not 16 kHz headset mode) and make it the default sink.
# Best effort: the app still starts (text replies) if the speaker is off.
mac=$(bluetoothctl devices Connected 2>/dev/null | awk '{print $2; exit}')
[ -n "$mac" ] || { for m in $(bluetoothctl devices Paired | awk '{print $2}'); do bluetoothctl connect "$m" >/dev/null 2>&1 && { mac=$m; break; }; done; }
[ -n "$mac" ] || { echo "bt_speaker: no Bluetooth speaker connected"; exit 0; }
card="bluez_card.${mac//:/_}"
for i in 1 2 3 4 5; do pactl list cards short | grep -q "$card" && break; sleep 1; done
pactl set-card-profile "$card" a2dp-sink 2>/dev/null || true
sink=$(pactl list short sinks | awk -v m="${mac//:/_}" '$2 ~ "bluez_output." m {print $2; exit}')
[ -n "$sink" ] && pactl set-default-sink "$sink" && echo "bt_speaker: default sink $sink"
