# Raspberry Pi 5 setup

This is the configuration used for the measurements in [`reports/Sorti_RaspberryPi5_Test_Report.pdf`](reports/Sorti_RaspberryPi5_Test_Report.pdf).

| Item | Used for the reported run |
|---|---|
| Board | Raspberry Pi 5, 8 GB, official 27 W USB-C supply, active cooler |
| OS | 64-bit Raspberry Pi OS (desktop), Python 3.13 |
| Camera | USB webcam (Logitech C270 in the test) |
| Microphone | USB microphone |
| Speaker | Bluetooth speaker (A2DP), or any ALSA / PipeWire output |
| Sorter (optional) | Arduino 2-servo sorter, connected over USB |
| Storage | microSD or SSD with at least 8 GB free |

Everything below is run as the desktop user, from the repository root.

## 1. System packages and Python environment

```bash
sudo apt update
sudo apt install -y git curl bzip2 python3-venv python3-dev ffmpeg alsa-utils pulseaudio-utils v4l-utils

git clone https://github.com/dharshan188/arm_ps2.git ~/sorti
cd ~/sorti

python3 -m venv ~/ps2_et
~/ps2_et/bin/pip install --upgrade pip
~/ps2_et/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
~/ps2_et/bin/pip install -r requirements.txt
```

- **Keep torch at 2.12.x.** ExecuTorch 1.3.1 is built against it. A newer torch fails with `undefined symbol ... materialize_cow_storage`.
- **Install torch from the CPU index** shown above.
- **Run Python as `env -u PYTHONPATH ~/ps2_et/bin/python`** so a ROS or other workspace on `PYTHONPATH` can't leak in.

Quick check of the vision runtime (also installs `v4l-utils` and lists cameras):

```bash
bash pi/setup_vision.sh
```

## 2. Speech, VAD and translation models

```bash
./download_models.sh        # about 2.3 GB, re-runnable
```

## 3. Hardware

### USB webcam

```bash
v4l2-ctl --list-devices      # use the FIRST /dev/videoN of the USB camera block
```

The Pi 5 exposes many internal `/dev/video*` nodes, so the USB camera is often not `/dev/video0`. Pass its number with `--camera N`.

### USB microphone

```bash
arecord -l                                    # find the card
arecord -f S16_LE -r 16000 -c 1 -d 3 /tmp/t.wav && aplay /tmp/t.wav
```

- Start the server with `--mic <part of the PulseAudio source name>`, or force the ALSA device: `SORTI_MIC_DEV=plughw:CARD=Device,DEV=0`.
- In the reported run, the USB mic measured 0.248 RMS room noise at 60 % gain and 0.045 at its lowest gain, so the lowest gain was used.
- Do not pick the Bluetooth speaker's own microphone. Its headset profile drops playback to 16 kHz.

### Bluetooth speaker

```bash
bluetoothctl       # power on ; agent on ; scan on ; pair XX:.. ; trust XX:.. ; connect XX:.. ; quit
bash pi/bt_speaker.sh          # A2DP profile + default sink
speaker-test -t wav -c 2 -l 1
```

- Leave `SORTI_SPK_DEV` unset for Bluetooth: playback goes to the default PipeWire sink.
- Run the app as the logged-in desktop user, because PipeWire is a per-user service. Over SSH you may need `export XDG_RUNTIME_DIR=/run/user/$(id -u)`.

### Arduino sorter (optional)

- Plug the Arduino in over USB. The server finds `/dev/serial/by-id/*`, `/dev/ttyACM*` or `/dev/ttyUSB*` by itself, or you can set `SORTI_SERIAL=/dev/ttyACM0`.
- The protocol is described in [`ARCHITECTURE.md`](ARCHITECTURE.md#5-arduino-sorter).
- Power the servos from a separate 5 V supply. During testing the board once dropped off USB while the servos were moving.

## 4. Run

```bash
cd ~/sorti
lap/run.sh --host 0.0.0.0 --camera <N> --mic usb             # with speaker
lap/run.sh --host 0.0.0.0 --camera <N> --no-speaker          # text replies, videos muted
```

Open `http://<pi-ip>:8000` from another device, or `http://127.0.0.1:8000` on the Pi. Stop with Ctrl+C. When the server runs in the background, stop it by PID, not with `pkill -f server.py`.

Vision-only preview, with no audio, on port 8080:

```bash
bash pi/start_vision.sh          # foreground;  --bg runs it in the background;  CAM=2 / PORT=8090 to override
bash pi/stop_vision.sh
```

### Autostart (optional)

The reported run used a systemd **user** service, which starts at login and therefore has access to the user's PipeWire session. A minimal unit:

```ini
# ~/.config/systemd/user/sorti-app.service
[Unit]
Description=Sorti waste-sorting assistant
After=graphical-session.target

[Service]
WorkingDirectory=%h/sorti/lap
ExecStartPre=-/bin/bash %h/sorti/pi/bt_speaker.sh
ExecStart=/bin/bash %h/sorti/lap/run.sh --host 0.0.0.0 --no-speaker
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now sorti-app.service
journalctl --user -u sorti-app -f
```

Adjust the `run.sh` flags (`--camera`, `--mic`, or drop `--no-speaker`) to match your hardware.

## 5. Llama 3.2 1B (optional)

Without Llama, Sorti answers every bin, recyclable, hazardous and where-to-give question, and the fixed demo questions. Llama adds answers to open questions.

**Pi route (used for the reported numbers): the public ExecuTorch SpinQuant export**

```bash
~/ps2_et/bin/hf download executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET --local-dir llama/_spin
ls llama/_spin                         # one .pte and tokenizer.model
cp llama/_spin/*.pte llama/llama3_2_1b_spinquant_int4_xnnpack.pte
mkdir -p llama/checkpoint && cp llama/_spin/tokenizer.model llama/checkpoint/tokenizer.model
```

Compare the downloaded files with the SHA-256 values published on the model page. The model is distributed under the Llama 3.2 Community License.

**Alternative: export Meta's QLoRA INT4 checkpoint yourself.** Accept the licence at `meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8`, then:

```bash
~/ps2_et/bin/hf auth login
llama/setup_llama.sh                   # about 6 min, about 6.5 GB free RAM (run on a laptop and copy the .pte if needed)
```

Test it:

```bash
cd llama && env -u PYTHONPATH ~/ps2_et/bin/python chat.py --pte llama3_2_1b_spinquant_int4_xnnpack.pte "How can I reduce plastic at home?"
```

The server uses the QLoRA `.pte` if present, otherwise the SpinQuant one. `SORTI_LLAMA_PTE` overrides both. Llama runs in a separate worker process and loads on the first open question (or in the background once a language is chosen).

## 6. Verify

```bash
env -u PYTHONPATH ~/ps2_et/bin/python -m unittest vision.test_multi lap.test_multi_flow
env -u PYTHONPATH ~/ps2_et/bin/python scripts/eval_testset.py
curl -s -X POST localhost:8000/api/ask/text -H 'Content-Type: application/json' -d '{"text":"which bin for a battery"}'
curl -s localhost:8000/api/state | python3 -m json.tool | grep -E 'ram_mb|ram_avail_mb'
vcgencmd measure_temp; vcgencmd get_throttled          # 0x0 = no under-voltage or throttling
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `undefined symbol ... materialize_cow_storage` | Reinstall `torch==2.12.1` from the CPU index. |
| `Missing operator llama::custom_sdpa.out` | Run Llama through `chat.py` or the app, which load the ExecuTorch custom ops first. |
| Strange imports from ROS or other packages | Run with `env -u PYTHONPATH` (`run.sh` already does this). |
| No camera image | Use the right `--camera N` from `v4l2-ctl --list-devices`. At boot the webcam can enumerate late, and the server keeps retrying. |
| Sorti hears nothing | Check `pactl list short sources`, pass `--mic <substring>`, and unmute the source. |
| No sound over Bluetooth | Reconnect with `bluetoothctl connect XX:..`, run `pi/bt_speaker.sh`, and run the app as the desktop user. |
| Port 8000 busy | `lap/run.sh --port 8010` |
| Harmless warnings | torchao `_C_mxfp8` / cutlass, cpuinfo `midr`, Qt font messages |
