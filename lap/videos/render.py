#!/usr/bin/env python3
"""Render code-drawn explainer videos from JSON specs (videos/specs/*.json), ~40 s each.

    PY="env -u PYTHONPATH $HOME/ps2_et/bin/python"
    $PY render.py specs/plastic_waste__bags.json   # one
    $PY render.py all                                # every spec not yet rendered (skips done ones)

Spec format: see specs/README.md. Output: videos/<cls>/<slug>.mp4 (silent) + <slug>_{en,hi,ta}.wav + .jpg,
registered in videos/index.json under <cls>. Frames hold only pictures and numbers; words are in the
voice-over and the app's subtitle bar.
"""
import glob, json, math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from explainer import (W, H, FPS, INK, BLUE_BIN, GREEN_BIN, RED_BIN, ease, stickman, bottle, bin_, cow, fish,
                       big_number, card)

GREY_BIN = (150, 158, 165)
BEATS = ["intro", "harm", "who", "dispose", "prevent"]
BG = {"intro": (255, 236, 179), "harm": (200, 228, 245), "who": (246, 241, 232), "dispose": (232, 240, 248),
      "prevent": (232, 245, 236)}
LW = 4


# ------------------------------------------------------------------ item kit: draw(d, x, y_base, s, t)
def _box(d, x, y, w, h, fill, r=8):
    d.rounded_rectangle([x - w / 2, y - h, x + w / 2, y], radius=r, fill=fill, outline=INK, width=LW)


def i_bottle(d, x, y, s, t): bottle(d, x, y, s)
def i_glass_bottle(d, x, y, s, t): bottle(d, x, y, s, color=(110, 180, 120))
def i_steel_bottle(d, x, y, s, t):
    _box(d, x, y, 60 * s, 170 * s, (185, 192, 200), 18); _box(d, x, y - 170 * s, 40 * s, 26 * s, (120, 128, 136), 4)
def i_bag(d, x, y, s, t):
    d.polygon([(x - 60 * s, y - 130 * s), (x + 60 * s, y - 130 * s), (x + 75 * s, y), (x - 75 * s, y)], fill=(240, 244, 248), outline=INK)
    d.arc([x - 50 * s, y - 190 * s, x - 5 * s, y - 110 * s], 180, 360, fill=INK, width=LW)
    d.arc([x + 5 * s, y - 190 * s, x + 50 * s, y - 110 * s], 180, 360, fill=INK, width=LW)
def i_cloth_bag(d, x, y, s, t):
    _box(d, x, y, 150 * s, 150 * s, (214, 186, 140), 16); d.arc([x - 45 * s, y - 220 * s, x + 45 * s, y - 120 * s], 180, 360, fill=INK, width=LW + 2)
def i_cup(d, x, y, s, t):
    d.polygon([(x - 45 * s, y - 120 * s), (x + 45 * s, y - 120 * s), (x + 32 * s, y), (x - 32 * s, y)], fill=(255, 255, 255), outline=INK)
    d.rectangle([x - 50 * s, y - 132 * s, x + 50 * s, y - 120 * s], fill=(230, 80, 70), outline=INK)
def i_steel_cup(d, x, y, s, t):
    d.polygon([(x - 45 * s, y - 120 * s), (x + 45 * s, y - 120 * s), (x + 35 * s, y), (x - 35 * s, y)], fill=(190, 196, 204), outline=INK)
def i_wrapper(d, x, y, s, t):
    d.polygon([(x - 80 * s, y - 70 * s), (x - 60 * s, y - 55 * s), (x - 60 * s, y - 15 * s), (x - 80 * s, y),
               (x + 80 * s, y), (x + 60 * s, y - 15 * s), (x + 60 * s, y - 55 * s), (x + 80 * s, y - 70 * s)], fill=(250, 200, 40), outline=INK)
    d.ellipse([x - 25 * s, y - 55 * s, x + 25 * s, y - 15 * s], fill=(230, 80, 70))
def i_milk_packet(d, x, y, s, t):
    d.rounded_rectangle([x - 55 * s, y - 140 * s, x + 55 * s, y], radius=int(20 * s), fill=(240, 248, 255), outline=INK, width=LW)
    d.rectangle([x - 55 * s, y - 95 * s, x + 55 * s, y - 60 * s], fill=(60, 120, 220))
def i_straw(d, x, y, s, t):
    d.line([(x - 10 * s, y), (x + 20 * s, y - 160 * s)], fill=(230, 80, 70), width=int(14 * s)); d.line([(x + 20 * s, y - 160 * s), (x + 50 * s, y - 175 * s)], fill=(230, 80, 70), width=int(14 * s))
def i_container(d, x, y, s, t):
    _box(d, x, y, 170 * s, 70 * s, (250, 250, 250), 10); d.rectangle([x - 92 * s, y - 84 * s, x + 92 * s, y - 70 * s], fill=(60, 60, 70))
def i_tiffin(d, x, y, s, t):
    for k in range(3): _box(d, x, y - k * 55 * s, 120 * s, 50 * s, (190, 196, 204), 12)
    d.arc([x - 40 * s, y - 210 * s, x + 40 * s, y - 150 * s], 180, 360, fill=INK, width=LW + 2)
def i_can(d, x, y, s, t):
    _box(d, x, y, 70 * s, 130 * s, (220, 60, 60), 10); d.rectangle([x - 35 * s, y - 85 * s, x + 35 * s, y - 50 * s], fill=(250, 250, 250))
def i_tin(d, x, y, s, t):
    _box(d, x, y, 100 * s, 120 * s, (200, 200, 205), 6); d.rectangle([x - 50 * s, y - 90 * s, x + 50 * s, y - 40 * s], fill=(240, 170, 60))
def i_foil(d, x, y, s, t):
    d.polygon([(x - 70 * s, y), (x - 50 * s, y - 60 * s), (x - 10 * s, y - 80 * s), (x + 40 * s, y - 65 * s), (x + 70 * s, y - 10 * s)], fill=(210, 214, 220), outline=INK)
def i_battery(d, x, y, s, t):
    _box(d, x, y, 60 * s, 150 * s, (40, 40, 45), 10); d.rectangle([x - 30 * s, y - 150 * s, x + 30 * s, y - 110 * s], fill=(240, 170, 40))
    d.rectangle([x - 12 * s, y - 165 * s, x + 12 * s, y - 150 * s], fill=(180, 180, 185), outline=INK)
def i_rechargeable(d, x, y, s, t):
    i_battery(d, x, y, s, t); d.arc([x - 70 * s, y - 120 * s, x + 70 * s, y + 20 * s], 200, 340, fill=GREEN_BIN, width=int(10 * s))
def i_button_cell(d, x, y, s, t):
    d.ellipse([x - 50 * s, y - 100 * s, x + 50 * s, y], fill=(200, 204, 210), outline=INK, width=LW)
    d.ellipse([x - 32 * s, y - 82 * s, x + 32 * s, y - 18 * s], outline=INK, width=2)
def i_phone(d, x, y, s, t):
    d.rounded_rectangle([x - 50 * s, y - 180 * s, x + 50 * s, y], radius=int(14 * s), fill=(40, 44, 52), outline=INK, width=LW)
    d.rectangle([x - 40 * s, y - 165 * s, x + 40 * s, y - 25 * s], fill=(120, 170, 220))
    d.line([(x - 30 * s, y - 150 * s), (x + 20 * s, y - 60 * s)], fill=(250, 250, 250), width=3)
def i_charger(d, x, y, s, t):
    _box(d, x, y - 60 * s, 70 * s, 60 * s, (250, 250, 250), 8)
    d.line([(x, y - 60 * s), (x - 40 * s, y - 20 * s), (x + 30 * s, y), (x + 80 * s, y - 30 * s)], fill=INK, width=int(6 * s), joint="curve")
    for k in (-14, 14): d.rectangle([x + k * s - 4 * s, y - 140 * s, x + k * s + 4 * s, y - 120 * s], fill=(160, 160, 165))
def i_laptop(d, x, y, s, t):
    d.rectangle([x - 110 * s, y - 150 * s, x + 110 * s, y - 20 * s], fill=(40, 44, 52), outline=INK, width=LW)
    d.rectangle([x - 98 * s, y - 140 * s, x + 98 * s, y - 30 * s], fill=(120, 170, 220))
    d.polygon([(x - 130 * s, y - 20 * s), (x + 130 * s, y - 20 * s), (x + 145 * s, y), (x - 145 * s, y)], fill=(170, 174, 180), outline=INK)
def i_earphones(d, x, y, s, t):
    d.line([(x - 50 * s, y - 120 * s), (x, y - 40 * s), (x + 50 * s, y - 120 * s)], fill=INK, width=int(5 * s))
    for k in (-1, 1): d.ellipse([x + k * 50 * s - 16 * s, y - 136 * s, x + k * 50 * s + 16 * s, y - 104 * s], fill=(250, 250, 250), outline=INK, width=LW)
    d.line([(x, y - 40 * s), (x, y)], fill=INK, width=int(5 * s))
def i_remote(d, x, y, s, t):
    d.rounded_rectangle([x - 32 * s, y - 170 * s, x + 32 * s, y], radius=int(14 * s), fill=(60, 64, 72), outline=INK, width=LW)
    for r in range(4):
        for c in (-1, 1): d.ellipse([x + c * 14 * s - 6 * s, y - 140 * s + r * 30 * s, x + c * 14 * s + 6 * s, y - 128 * s + r * 30 * s], fill=(200, 200, 205))
def i_bulb(d, x, y, s, t):
    d.ellipse([x - 55 * s, y - 175 * s, x + 55 * s, y - 65 * s], fill=(255, 236, 140), outline=INK, width=LW)
    _box(d, x, y, 50 * s, 60 * s, (180, 180, 185), 4)
def i_cfl(d, x, y, s, t):
    for k in (-24, 0, 24): d.rounded_rectangle([x + k * s - 9 * s, y - 190 * s, x + k * s + 9 * s, y - 60 * s], radius=int(9 * s), fill=(250, 250, 245), outline=INK, width=3)
    _box(d, x, y, 80 * s, 60 * s, (240, 240, 240), 8)
def i_led_bulb(d, x, y, s, t):
    i_bulb(d, x, y, s, t); d.rectangle([x - 40 * s, y - 120 * s, x + 40 * s, y - 112 * s], fill=GREEN_BIN)
def i_tube(d, x, y, s, t):
    d.rounded_rectangle([x - 160 * s, y - 40 * s, x + 160 * s, y], radius=int(20 * s), fill=(250, 250, 245), outline=INK, width=LW)
    for k in (-1, 1): d.rectangle([x + k * 160 * s - 10 * s, y - 36 * s, x + k * 160 * s + 10 * s, y - 4 * s], fill=(170, 170, 175))
def i_newspaper(d, x, y, s, t):
    d.polygon([(x - 90 * s, y - 120 * s), (x + 90 * s, y - 130 * s), (x + 95 * s, y), (x - 85 * s, y + 8 * s)], fill=(245, 242, 232), outline=INK)
    for k in range(5): d.line([(x - 70 * s, y - 100 * s + k * 20 * s), (x + 70 * s, y - 106 * s + k * 20 * s)], fill=(150, 150, 150), width=3)
def i_box(d, x, y, s, t):
    _box(d, x, y, 170 * s, 120 * s, (205, 160, 105), 4); d.line([(x, y - 120 * s), (x, y)], fill=(170, 125, 75), width=int(10 * s))
def i_notebook(d, x, y, s, t):
    _box(d, x, y, 120 * s, 160 * s, (70, 130, 200), 6)
    for k in range(6): d.ellipse([x - 66 * s, y - 150 * s + k * 24 * s, x - 54 * s, y - 138 * s + k * 24 * s], fill=(250, 250, 250), outline=INK)
def i_paper_cup(d, x, y, s, t):
    d.polygon([(x - 45 * s, y - 120 * s), (x + 45 * s, y - 120 * s), (x + 32 * s, y), (x - 32 * s, y)], fill=(245, 240, 225), outline=INK)
def i_paper_bag(d, x, y, s, t):
    _box(d, x, y, 120 * s, 160 * s, (200, 160, 110), 4); d.polygon([(x - 60 * s, y - 160 * s), (x + 60 * s, y - 160 * s), (x + 45 * s, y - 180 * s), (x - 45 * s, y - 180 * s)], fill=(185, 145, 95), outline=INK)
def i_jar(d, x, y, s, t):
    d.rounded_rectangle([x - 55 * s, y - 130 * s, x + 55 * s, y], radius=int(18 * s), fill=(200, 230, 225), outline=INK, width=LW)
    d.rectangle([x - 45 * s, y - 155 * s, x + 45 * s, y - 130 * s], fill=(230, 80, 70), outline=INK)
    d.rectangle([x - 45 * s, y - 80 * s, x + 45 * s, y - 40 * s], fill=(255, 220, 120))
def i_broken_glass(d, x, y, s, t):
    for k, (dx, h) in enumerate([(-60, 60), (-15, 90), (35, 50), (70, 35)]):
        d.polygon([(x + dx * s - 25 * s, y), (x + dx * s, y - h * s), (x + dx * s + 22 * s, y)], fill=(170, 220, 215), outline=INK)
def i_peel(d, x, y, s, t):
    d.polygon([(x, y - 60 * s), (x - 70 * s, y), (x - 40 * s, y - 50 * s)], fill=(250, 215, 70), outline=INK)
    d.polygon([(x, y - 60 * s), (x + 70 * s, y), (x + 40 * s, y - 50 * s)], fill=(250, 215, 70), outline=INK)
    d.polygon([(x - 15 * s, y - 60 * s), (x + 15 * s, y - 60 * s), (x, y - 130 * s)], fill=(250, 215, 70), outline=INK)
def i_food(d, x, y, s, t):
    d.chord([x - 100 * s, y - 80 * s, x + 100 * s, y + 40 * s], 0, 180, fill=(250, 250, 250), outline=INK, width=LW)
    for k in range(7): d.ellipse([x - 70 * s + k * 20 * s, y - 40 * s - (k % 2) * 12 * s, x - 52 * s + k * 20 * s, y - 24 * s - (k % 2) * 12 * s], fill=(250, 240, 200), outline=INK)
def i_apple(d, x, y, s, t):
    d.ellipse([x - 25 * s, y - 120 * s, x + 25 * s, y], fill=(250, 240, 210), outline=INK, width=LW)
    d.chord([x - 45 * s, y - 130 * s, x + 45 * s, y - 80 * s], 180, 360, fill=(220, 60, 60), outline=INK)
    d.chord([x - 45 * s, y - 45 * s, x + 45 * s, y + 5 * s], 0, 180, fill=(220, 60, 60), outline=INK)
def i_leaves(d, x, y, s, t):
    for k, dx in enumerate((-60, 0, 60)):
        d.ellipse([x + dx * s - 30 * s, y - 70 * s - (k % 2) * 30 * s, x + dx * s + 30 * s, y - (k % 2) * 30 * s], fill=(80, 170, 80), outline=INK, width=3)
def i_tea(d, x, y, s, t):
    d.ellipse([x - 70 * s, y - 50 * s, x + 70 * s, y], fill=(110, 75, 45), outline=INK, width=3)
def i_flowers(d, x, y, s, t):
    for k, dx in enumerate((-50, 0, 50)):
        d.line([(x + dx * s, y), (x + dx * s, y - 110 * s)], fill=(60, 140, 60), width=int(6 * s))
        cx, cy = x + dx * s, y - 120 * s
        for a in range(5):
            d.ellipse([cx + 22 * s * math.cos(a * 1.256) - 14 * s, cy + 22 * s * math.sin(a * 1.256) - 14 * s,
                       cx + 22 * s * math.cos(a * 1.256) + 14 * s, cy + 22 * s * math.sin(a * 1.256) + 14 * s], fill=(250, 150, 40), outline=INK)
def i_compost_bin(d, x, y, s, t):
    _box(d, x, y, 150 * s, 170 * s, (110, 80, 50), 10)
    for k in range(3): d.line([(x - 60 * s, y - 130 * s + k * 40 * s), (x + 60 * s, y - 130 * s + k * 40 * s)], fill=(80, 55, 35), width=int(6 * s))
    i_leaves(d, x, y - 170 * s, 0.6 * s, t)
def i_shirt(d, x, y, s, t):
    d.polygon([(x - 50 * s, y - 160 * s), (x - 110 * s, y - 120 * s), (x - 85 * s, y - 85 * s), (x - 55 * s, y - 105 * s),
               (x - 55 * s, y), (x + 55 * s, y), (x + 55 * s, y - 105 * s), (x + 85 * s, y - 85 * s), (x + 110 * s, y - 120 * s),
               (x + 50 * s, y - 160 * s), (x + 20 * s, y - 145 * s), (x - 20 * s, y - 145 * s)], fill=(80, 150, 210), outline=INK)
def i_jeans(d, x, y, s, t):
    d.polygon([(x - 60 * s, y - 170 * s), (x + 60 * s, y - 170 * s), (x + 70 * s, y), (x + 15 * s, y), (x, y - 110 * s),
               (x - 15 * s, y), (x - 70 * s, y)], fill=(60, 90, 160), outline=INK)
def i_saree(d, x, y, s, t):
    for k in range(4): d.rounded_rectangle([x - 90 * s, y - 40 * s - k * 30 * s, x + 90 * s, y - k * 30 * s], radius=int(10 * s),
                                           fill=[(220, 60, 110), (250, 190, 40), (220, 60, 110), (250, 190, 40)][k], outline=INK, width=3)
def i_donate_box(d, x, y, s, t):
    i_box(d, x, y, s, t); d.ellipse([x - 25 * s, y - 95 * s, x + 25 * s, y - 50 * s], fill=(236, 90, 79))
def i_wrench(d, x, y, s, t):
    d.line([(x - 70 * s, y), (x + 40 * s, y - 110 * s)], fill=(150, 156, 164), width=int(22 * s))
    d.ellipse([x + 20 * s, y - 150 * s, x + 80 * s, y - 90 * s], fill=(150, 156, 164), outline=INK, width=LW)
    d.ellipse([x + 38 * s, y - 132 * s, x + 62 * s, y - 108 * s], fill=BG["prevent"])
def i_cable(d, x, y, s, t):
    pts = [(x - 90 * s + k * 15 * s, y - 40 * s - 30 * s * math.sin(k * 0.9)) for k in range(13)]
    d.line(pts, fill=INK, width=int(8 * s), joint="curve")


ITEMS = {k[2:]: v for k, v in globals().items() if k.startswith("i_")}


# ------------------------------------------------------------------ "who it affects" kit
def v_cow(d, cx, cy, s, t): cow(d, cx - 30, cy, 0.85)
def v_fish(d, cx, cy, s, t): fish(d, cx, cy - 60, 0.95, t)
def v_worker(d, cx, cy, s, t): stickman(d, cx, cy + 20, 0.62, t, pose="sad", talking=False, vest=(255, 140, 0))
def v_child(d, cx, cy, s, t): stickman(d, cx, cy + 20, 0.45, t, pose="sad", talking=False)
def v_person(d, cx, cy, s, t): stickman(d, cx, cy + 20, 0.6, t, pose="sad", talking=False)
def v_bird(d, cx, cy, s, t):
    d.ellipse([cx - 50, cy - 110, cx + 40, cy - 50], fill=(120, 160, 200), outline=INK, width=4)
    d.ellipse([cx + 20, cy - 140, cx + 70, cy - 90], fill=(120, 160, 200), outline=INK, width=4)
    d.polygon([(cx + 68, cy - 118), (cx + 95, cy - 110), (cx + 68, cy - 102)], fill=(250, 180, 40), outline=INK)
    d.ellipse([cx + 42, cy - 125, cx + 50, cy - 117], fill=INK)
def v_dog(d, cx, cy, s, t):
    d.rounded_rectangle([cx - 70, cy - 100, cx + 40, cy - 55], radius=20, fill=(200, 150, 90), outline=INK, width=4)
    for lx in (-55, -30, 10, 30): d.line([(cx + lx, cy - 60), (cx + lx, cy)], fill=INK, width=6)
    d.ellipse([cx + 20, cy - 140, cx + 80, cy - 85], fill=(200, 150, 90), outline=INK, width=4)
    d.ellipse([cx + 55, cy - 122, cx + 63, cy - 114], fill=INK)
def v_river(d, cx, cy, s, t):
    for k in range(3):
        pts = [(cx - 100 + i * 20, cy - 60 - k * 40 + 10 * math.sin(i * 0.8 + t * 3)) for i in range(11)]
        d.line(pts, fill=(60, 140, 220), width=10, joint="curve")
def v_soil(d, cx, cy, s, t):
    d.rectangle([cx - 100, cy - 90, cx + 100, cy], fill=(140, 98, 60))
    d.line([(cx - 20, cy - 90), (cx - 20, cy - 150)], fill=(60, 140, 60), width=6)
    d.ellipse([cx - 45, cy - 160, cx - 20, cy - 135], fill=(150, 170, 80), outline=INK)
def v_farmer(d, cx, cy, s, t):
    stickman(d, cx, cy + 20, 0.6, t, pose="sad", talking=False)
    d.polygon([(cx - 40, cy - 165), (cx + 40, cy - 165), (cx, cy - 195)], fill=(230, 200, 120), outline=INK)
def v_air(d, cx, cy, s, t):
    for k in range(4):
        r = 30 + 8 * math.sin(t * 2 + k)
        d.ellipse([cx - 70 + k * 35 - r, cy - 110 - k * 15 - r, cx - 70 + k * 35 + r, cy - 110 - k * 15 + r], fill=(150, 150, 155))


VICTIMS = {k[2:]: v for k, v in globals().items() if k.startswith("v_")}
BINS = {"blue": BLUE_BIN, "green": GREEN_BIN, "red": RED_BIN, "ewaste": GREY_BIN}


def item(spec_item, d, x, y, s, t):
    ITEMS.get(spec_item, i_bottle)(d, x, y, s, t)


# ------------------------------------------------------------------ scenes
def sc_intro(d, t, D, sp):
    d.rectangle([0, 560, W, H], fill=(236, 214, 160))
    x = 200 + 380 * ease(t / 1.3)
    head = stickman(d, x, 600, 1.25, t, pose="wave" if t > 1.2 else "talk")
    if t < D * 0.55:
        item(sp["item"], d, x + 10, -80 + (head[1] - 60) * ease((t - 0.8) / (D * 0.45)), 0.7, t)
    else:
        k = t - D * 0.55
        item(sp["item"], d, x + 160 + 60 * k, 600 - max(0, 40 - 260 * k), 0.7, t)
        for i in range(5):
            a = t * 5 + i * 1.256
            d.regular_polygon((head[0] + 70 * math.cos(a), head[1] - 50 + 18 * math.sin(a), 10), 5, fill=(255, 196, 0))
    if t > 1.3:
        bx = x + 140
        d.rounded_rectangle([bx, 150, bx + 230, 280], radius=40, fill=(255, 255, 255), outline=INK, width=5)
        d.polygon([(bx + 40, 270), (bx + 10, 330), (bx + 90, 275)], fill=(255, 255, 255), outline=INK)
        big_number(d, "?!", bx + 115, 215, 90, (236, 90, 79))


def counter(d, st, k, cx, cy, size=170, color=INK):
    v = st["value"] * k
    dec = st.get("decimals", 0)
    txt = f"{v:,.{dec}f}" if dec else f"{int(round(v)):,}"
    big_number(d, st.get("prefix", "") + txt + st.get("suffix", ""), cx, cy, size, color)
    if st.get("unit"):
        big_number(d, st["unit"], cx, cy + size * 0.75, 48, (90, 96, 105), maxw=2 * min(cx, W - cx) - 50)


def sc_harm(d, t, D, sp):
    d.rectangle([0, 430, W, H], fill=(140, 98, 60))
    for i in range(6):
        d.ellipse([80 + i * 200, 480 + (i % 2) * 90, 120 + i * 200, 500 + (i % 2) * 90], fill=(110, 76, 46))
    item(sp["item"], d, 430, 440 + 110 * ease(t / D), 1.3, t)
    stats = sp["beats"]["harm"].get("stats") or []
    if stats:
        n = len(stats); idx = min(n - 1, int(t / (D / n))); lt = t - idx * D / n
        counter(d, stats[idx], ease(lt / (D / n * 0.7)), 930, 230)
    stickman(d, 150, 650, 0.8, t, pose="sad" if t > D * 0.7 else "talk")


def sc_who(d, t, D, sp):
    vs = (sp["beats"]["who"].get("victims") or ["person", "child"])[:4]
    n = len(vs); gap = (W - 80) / n
    for i, v in enumerate(vs):
        x = 40 + i * gap + gap / 2 - 120; y = 120
        ti = t - i * (D * 0.18)
        if ti < 0:
            continue
        pop = ease(ti / 0.35); cw, ch = 240 * pop, 300 * pop
        card(d, x + 120 - cw / 2, y + 150 - ch / 2, cw, ch)
        if pop < 0.95:
            continue
        VICTIMS.get(v, v_person)(d, x + 120, y + 250, 1, t)
        r = 22 + 6 * math.sin(t * 6)
        d.ellipse([x + 205 - r, y + 15 - r, x + 205 + r, y + 15 + r], fill=(236, 90, 79))
        big_number(d, "!", x + 205, y + 18, 36, (255, 255, 255))
    stickman(d, 640, 700, 0.7, t, pose="talk")


def sc_dispose(d, t, D, sp):
    b = sp["beats"]["dispose"]
    d.rectangle([0, 560, W, H], fill=(205, 222, 235))
    color = BINS.get(b.get("bin", "blue"), BLUE_BIN)
    lid = ease((t - D * 0.45) / 0.4) * (1 - ease((t - D * 0.7) / 0.3))
    bin_(d, 860, 600, color, lid=lid)
    if b.get("bin") == "ewaste":   # e-waste: a collection box with a lightning mark, not a household bin
        d.polygon([(860, 450), (835, 510), (860, 505), (845, 560), (890, 490), (865, 495)], fill=(255, 210, 0), outline=INK)
    steps = b.get("steps") or []
    p = t / D
    prep_end = 0.42 if steps else 0.15
    if p < prep_end:
        k = int(p / prep_end * max(1, len(steps))) if steps else 0
        stp = steps[min(k, len(steps) - 1)] if steps else ""
        item(sp["item"], d, 330, 560, 1.15, t)
        if stp == "rinse":
            for i in range(8):
                yy = (t * 400 + i * 60) % 260 + 220
                d.ellipse([300 + (i % 3) * 25, yy, 312 + (i % 3) * 25, yy + 18], fill=(80, 160, 230))
        elif stp == "squash":
            d.rounded_rectangle([250, 300 + 80 * math.sin(t * 6) ** 2, 410, 330 + 80 * math.sin(t * 6) ** 2], radius=10, fill=(120, 120, 120))
        elif stp == "tape":
            d.rectangle([290, 380, 370, 400], fill=(250, 220, 90), outline=INK)
        elif stp == "dry":
            big_number(d, "☀", 330, 250, 90, (250, 180, 30))
        elif stp in ("wrap", "box"):
            d.rectangle([240, 420, 420, 565], outline=(205, 160, 105), width=10)
    elif p < 0.62:
        k = ease((p - prep_end) / (0.62 - prep_end))
        bx = 330 + (860 - 330) * k; by = 560 - 380 * math.sin(math.pi * k) + 80 * k
        item(sp["item"], d, bx, by, 1.15 * (1 - 0.45 * k), t)
    st = b.get("stat")
    if st and p > 0.62:
        k = ease((p - 0.62) / 0.25)
        if st.get("suffix") == "%":
            cx, cy, r = 1110, 250, 110
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 220, 215))
            d.pieslice([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * st["value"] / 100 * k, fill=GREEN_BIN)
            counter(d, st, k, cx, cy + 175, 80, GREEN_BIN)
        else:
            counter(d, st, k, 1100, 230, 110, GREEN_BIN)
    stickman(d, 130, 640, 0.8, t, pose="talk")


def sc_prevent(d, t, D, sp):
    b = sp["beats"]["prevent"]
    p = t / D
    k = ease(p / 0.3)
    if k < 1:
        item(b.get("swap_from", sp["item"]), d, 300, 560, 1.1, t)
        d.line([(190, 330), (410, 590)], fill=RED_BIN, width=int(14 * k))
        d.line([(410, 330), (190, 590)], fill=RED_BIN, width=int(14 * k))
    else:
        item(b.get("swap_to", "cloth_bag"), d, 300, 560, 1.1, t)
        d.line([(420, 470), (445, 500), (490, 430)], fill=GREEN_BIN, width=12)
    if b.get("swap_to2") and p > 0.3:
        item(b["swap_to2"], d, 600, 560, 1.0, t)
    st = b.get("stat")
    if st and p > 0.45:
        counter(d, st, ease((p - 0.45) / 0.3), 1000, 240, 130, (236, 90, 79) if st.get("bad") else GREEN_BIN)
    stickman(d, 800 if b.get("swap_to2") else 700, 660, 0.75, t, pose="thumbs" if p > 0.75 else "talk")


SCENES = {"intro": sc_intro, "harm": sc_harm, "who": sc_who, "dispose": sc_dispose, "prevent": sc_prevent}


# ------------------------------------------------------------------ render one spec
_VOICES = {}


def render(spec_path):
    from build import tts_voices, synth
    sp = json.load(open(spec_path))
    cls, slug = sp["cls"], sp["slug"]
    out = os.path.join(HERE, cls); raw = os.path.join(out, "raw"); os.makedirs(raw, exist_ok=True)
    if not _VOICES:
        _VOICES.update(tts_voices())
    seg = {}
    for b in BEATS:
        for lang in ("en", "hi", "ta"):
            p = os.path.join(raw, f"{slug}_{b}_{lang}.wav")
            seg[b, lang] = (p, synth(_VOICES[lang], sp["beats"][b][lang], p, lang))
    dur = {b: max(seg[b, l][1] for l in ("en", "hi", "ta")) + 0.6 for b in BEATS}
    starts, acc = {}, 0.3
    for b in BEATS:
        starts[b] = acc; acc += dur[b]
    total = acc + 0.4
    for lang in ("en", "hi", "ta"):
        buf, sr = None, None
        for b in BEATS:
            with wave.open(seg[b, lang][0]) as w:
                sr = sr or w.getframerate(); y = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if buf is None:
                buf = np.zeros(int(total * sr), np.int16)
            i = int(starts[b] * sr); buf[i:i + len(y)] = y[:len(buf) - i]
            os.remove(seg[b, lang][0])
        with wave.open(os.path.join(out, f"{slug}_{lang}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(buf.tobytes())
    mp4 = os.path.join(out, f"{slug}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    poster = False
    for f in range(int(total * FPS)):
        t = f / FPS
        beat = max((b for b in BEATS if starts[b] <= t), key=lambda b: starts[b], default="intro")
        lt = t - starts[beat]
        img = Image.new("RGB", (W, H), BG[beat]); d = ImageDraw.Draw(img)
        SCENES[beat](d, lt, dur[beat], sp)
        for i, b in enumerate(BEATS):
            d.ellipse([W / 2 - 70 + i * 32, H - 34, W / 2 - 56 + i * 32, H - 20], fill=(243, 189, 63) if b == beat else (200, 200, 195))
        arr = np.asarray(img)
        fade = min(1.0, lt / 0.25)
        if fade < 1:
            arr = (arr.astype(np.float32) * fade + 255 * (1 - fade)).astype(np.uint8)
        ff.stdin.write(arr.tobytes())
        if beat == "harm" and not poster and lt > 2.5:
            Image.fromarray(arr).save(os.path.join(out, f"{slug}.jpg"), quality=85); poster = True
    ff.stdin.close(); ff.wait()
    register(sp)
    print(f"done {cls}/{slug}.mp4 {total:.1f}s", flush=True)


def register(sp):
    if os.environ.get("SORTI_NO_INDEX"):          # parallel renderer: index rebuilt later by `reindex`
        return
    cls, slug = sp["cls"], sp["slug"]
    p = os.path.join(HERE, "index.json")
    idx = json.load(open(p)) if os.path.exists(p) else {}
    vo = {l: " ".join(sp["beats"][b][l] for b in BEATS) for l in ("en", "hi", "ta")}
    entry = dict(id=slug, title=sp.get("title", {}), video=f"{cls}/{slug}.mp4",
                 vo_files={l: f"{cls}/{slug}_{l}.wav" for l in ("en", "hi", "ta")}, poster=f"{cls}/{slug}.jpg",
                 credit="Sorti (drawn in code)", page="", vo=vo, sources=sp.get("sources", []))
    idx[cls] = [c for c in idx.get(cls, []) if c["id"] != slug] + [entry]
    json.dump(idx, open(p, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "reindex":                          # register every spec whose video exists
        for s_ in sorted(glob.glob(os.path.join(HERE, "specs", "**", "*.json"), recursive=True)):
            sp_ = json.load(open(s_))
            if os.path.exists(os.path.join(HERE, sp_["cls"], sp_["slug"] + ".mp4")):
                register(sp_)
        print("reindexed"); sys.exit()
    specs = sorted(glob.glob(os.path.join(HERE, "specs", "*.json"))) if arg == "all" else [arg]
    for s in specs:
        sp = json.load(open(s))
        if arg == "all" and os.path.exists(os.path.join(HERE, sp["cls"], sp["slug"] + ".mp4")):
            continue
        try:
            render(s)
        except Exception as e:
            print("FAILED", s, repr(e), flush=True)
