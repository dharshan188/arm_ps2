#!/usr/bin/env python3
"""Code-drawn explainer video (no stock footage, no AI imagery): a stick-man narrator + flat illustrations.

    env -u PYTHONPATH ~/ps2_et/bin/python explainer.py plastic_waste

Writes videos/<class>/explainer.mp4 (silent 1280x720, scenes timed to the longest language) and
explainer_{en,hi,ta}.wav, and registers it in videos/index.json (first clip for that class).
The words live in the voice-over and the app's subtitle bar, so frames hold only numbers and pictures.

Stats used (plastic):
- ~450 years for a PET bottle to break down: NOAA / Woods Hole Sea Grant estimate.
- ~3.4 million tonnes of plastic waste per year in India: CPCB Annual Report 2019-20 (3.47 MT).
- 9% of plastic waste is recycled: OECD Global Plastics Outlook 2022.
- "more plastic than fish by weight by 2050": Ellen MacArthur Foundation 2016 projection.
"""
import json, math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W, H, FPS = 1280, 720, 24
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK, SKIN = (34, 40, 49), (255, 255, 255)
BLUE_BIN, GREEN_BIN, RED_BIN = (74, 149, 232), (63, 191, 111), (236, 90, 79)

SCRIPTS = {"plastic_waste": {
    "intro": {
        "en": "Hey you! Yes, you with the plastic bottle! Give me forty seconds and I'll turn you into a waste ninja.",
        "hi": "अरे तुम! हाँ हाँ, तुम, बोतल वाले! बस चालीस सेकंड दो, तुम्हें वेस्ट निंजा बना दूँगा।",
        "ta": "ஹே நீங்க! ஆமா, அந்த பாட்டில் வெச்சிருக்கீங்களே, நீங்க தான்! ஒரு நாப்பது செகண்ட் குடுங்க, உங்கள வேஸ்ட் நின்ஜா ஆக்கிடறேன்."},
    "harm": {
        "en": "One plastic bottle takes about four hundred and fifty years to break down. And India throws away around three point four million tonnes of plastic every year!",
        "hi": "एक प्लास्टिक बोतल को गलने में करीब साढ़े चार सौ साल लगते हैं। और इंडिया हर साल लगभग चौंतीस लाख टन प्लास्टिक कचरा फेंकता है!",
        "ta": "ஒரு பிளாஸ்டிக் பாட்டில் மக்கி போக கிட்டத்தட்ட நானூத்தி அம்பது வருஷம் ஆகும். நம்ம இந்தியா ஒவ்வொரு வருஷமும் முப்பத்தி நாலு லட்சம் டன் பிளாஸ்டிக் குப்பைய தூக்கி போடுது!"},
    "who": {
        "en": "Who pays for it? Cows eat our plastic bags, fish swallow tiny bits, and sanitation workers pull it out of drains by hand.",
        "hi": "भुगतता कौन है? गायें हमारी पन्नी खा जाती हैं, मछलियाँ छोटे टुकड़े निगल जाती हैं, और सफ़ाई वाले भैया नालियों से इसे हाथ से निकालते हैं।",
        "ta": "இதுல யாருக்கு கஷ்டம்? மாடுங்க நம்ம கவர சாப்பிடுது, மீனுங்க சின்ன துண்டுகள முழுங்குது, சாக்கடைல இருந்து இத கையால எடுக்குறது சுத்தம் பண்ற அண்ணன்ங்க தான்."},
    "dispose": {
        "en": "So rinse it, squash it, and drop it in the blue bin. So far, only nine per cent of plastic waste gets recycled.",
        "hi": "तो बोतल धोओ, दबाओ, और ब्लू बिन में डालो। अभी तक सिर्फ़ नौ परसेंट प्लास्टिक कचरा ही रीसायकल होता है।",
        "ta": "அதனால பாட்டில கழுவி, நசுக்கி, ப்ளூ பின்ல போடுங்க. இப்போ வரைக்கும் வெறும் ஒன்பது பர்சன்ட் பிளாஸ்டிக் தான் ரீசைக்கிள் ஆகுது."},
    "prevent": {
        "en": "Even better, carry a cloth bag and a steel bottle. Otherwise by twenty fifty, the sea could hold more plastic than fish. Deal? High five!",
        "hi": "और भी बढ़िया, कपड़े का थैला और स्टील की बोतल साथ रखो। वरना दो हज़ार पचास तक समंदर में मछलियों से ज़्यादा प्लास्टिक हो सकता है। डील? हाई फ़ाइव!",
        "ta": "இன்னும் சூப்பர்னா, துணிப்பை, ஸ்டீல் பாட்டில் கூடவே வெச்சுக்கோங்க. இல்லனா ரெண்டாயிரத்து அம்பதுல கடல்ல மீன விட பிளாஸ்டிக் அதிகமா இருக்கலாம். டீலா? ஹை ஃபைவ்!"},
}}
ORDER = ["intro", "harm", "who", "dispose", "prevent"]


def ease(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


def font(n):
    return ImageFont.truetype(FONT, n)


# ------------------------------------------------------------------ drawing kit
def stickman(d, x, y, s=1.0, t=0.0, pose="talk", talking=True, vest=None):
    """Round-jointed cartoon; feet stay at y, return the animated head center."""
    # Work in character units on a small supersampled tile, including face details.
    # This keeps the same stroke hierarchy on the tiny victim cards.
    left, top = math.floor(x - 105 * s), math.floor(y - 270 * s)
    right, bottom = math.ceil(x + 110 * s), math.ceil(y + 2 * s)
    aa = 3
    tile = Image.new("RGBA", ((right - left) * aa, (bottom - top) * aa))
    pen = ImageDraw.Draw(tile)
    unit = s * aa
    stroke = max(7.0, 2.8 / s)

    def point(p):
        return ((x - left + p[0] * s) * aa,
                (y - top + p[1] * s) * aa)

    def disc(p, radius, fill=INK):
        px, py = point(p); r = radius * unit
        pen.ellipse((px - r, py - r, px + r, py + r), fill=fill)

    def chain(points, width=stroke, fill=INK):
        pen.line([point(p) for p in points], fill=fill,
                 width=max(1, round(width * unit)))
        for p in points:
            disc(p, width / 2, fill)

    def extend(p, length, angle):
        return (p[0] + length * math.cos(angle),
                p[1] + length * math.sin(angle))

    # One slow four-second breathing loop; silent victims breathe more subtly.
    phase = t * math.tau / 4
    strength = 1.0 if talking else 0.45
    sway = 1.8 * math.sin(phase) * strength
    breath = 1.2 * math.sin(phase) * strength
    hip = (sway * 0.45, -101 + breath * 0.45)
    shoulder_y = -176 + breath
    shoulders = ((sway - 14, shoulder_y), (sway + 14, shoulder_y))
    head = (sway, -224 + breath)
    neck = (head[0], head[1] + 34 - 2)  # concealed overlap under head outline

    # Equal-length thighs/shins solve to fixed ankles; knees absorb the sway.
    for side in (-1, 1):
        ankle = (side * 28, -stroke / 2)
        dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
        reach = math.hypot(dx, dy)
        bend = math.sqrt(max(0, 52 ** 2 - (reach / 2) ** 2))
        knee = ((hip[0] + ankle[0]) / 2 + side * dy / reach * bend,
                (hip[1] + ankle[1]) / 2 - side * dx / reach * bend)
        chain([hip, knee, ankle])
        chain([ankle, (side * 38, ankle[1])])
    chain([hip, (sway, shoulder_y), neck])
    chain(shoulders)

    if vest:
        # Shoulder cutouts, V neckline, and a hem following the moving hip.
        outline = [(sway - 13, shoulder_y - 1), (sway, shoulder_y + 10),
                   (sway + 13, shoulder_y - 1), (sway + 21, shoulder_y + 11),
                   (hip[0] + 18, hip[1] - 4), (hip[0] - 18, hip[1] - 4),
                   (sway - 21, shoulder_y + 11)]
        pen.polygon([point(p) for p in outline], fill=vest)
        chain(outline + [outline[0]], 2.4)
        chain([(sway, shoulder_y + 10), (hip[0], hip[1] - 5)], 2.0)

    # Every pose uses two 42-unit arm segments attached to separate shoulders.
    gesture = math.sin(phase)
    left_angles = (2.15, 1.67)
    if pose == "sad":
        left_angles = (2.18, 1.76)
        right_angles = (math.pi - left_angles[0], math.pi - left_angles[1])
    elif pose == "wave":
        right_angles = (-0.38, -1.18 + 0.22 * math.sin(2 * phase))
    elif pose == "thumbs":
        right_angles = (0.60, -1.12)
    else:
        right_angles = (0.50 + gesture * 0.08, -0.43 + gesture * 0.14)

    hands = []
    for shoulder, angles in zip(shoulders, (left_angles, right_angles)):
        elbow = extend(shoulder, 42, angles[0])
        wrist = extend(elbow, 42, angles[1])
        chain([shoulder, elbow, wrist])
        hands.append(wrist)
    hand_radius = max(6.3, 3.0 / s)
    for i, wrist in enumerate(hands):
        if pose == "thumbs" and i == 1:
            # A short upright thumb above a compact round fist.
            a = (wrist[0] - 5, wrist[1] - 1)
            b = (wrist[0] - 5, wrist[1] - 12)
            chain([a, b], 7.5)
            chain([a, b], 2.5, SKIN)
            wrist = (wrist[0] + 1.5, wrist[1])
        radius = hand_radius + (1.2 if pose == "thumbs" and i == 1 else 0)
        disc(wrist, radius)
        disc(wrist, radius - 2.6, SKIN)

    disc(head, 34)
    disc(head, 34 - stroke, SKIN)
    eye_radius = max(3.3, 1.5 / s)
    for side in (-1, 1):
        disc((head[0] + side * 11, head[1] - 5), eye_radius)
        if pose == "sad":
            chain([(head[0] + side * 6, head[1] - 17),
                   (head[0] + side * 17, head[1] - 13)], 2.5)

    # A smoothly opening curved mouth keeps the expression and corners stable.
    opening = (0.5 - 0.5 * math.cos(t * math.tau * 2.5)) * 5 if talking else 0
    upper, lower = [], []
    for i in range(25):
        u = i / 24
        arch = math.sin(math.pi * u)
        curve = (4 - 4 * arch) if pose == "sad" else 5 * arch
        mx = head[0] - 10 + 20 * u
        my = head[1] + 12 + curve
        upper.append((mx, my - opening * arch * 0.25))
        lower.append((mx, my + opening * arch * 0.75))
    pen.polygon([point(p) for p in upper + lower[::-1]], fill=INK)
    chain(upper, 2.3)
    chain(lower, 2.3)

    tile = tile.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    d._image.paste(tile, (left, top), tile)
    return (x + head[0] * s, y + head[1] * s)

def bottle(d, x, y, s=1.0, squash=0.0, color=(150, 205, 240)):
    """Plastic bottle standing with its base at (x, y)."""
    h = 150 * s * (1 - 0.6 * squash); w = 52 * s * (1 + 0.35 * squash)
    d.rounded_rectangle([x - w / 2, y - h, x + w / 2, y], radius=int(14 * s), fill=color, outline=INK, width=max(2, int(4 * s)))
    d.rectangle([x - w / 2 + 3 * s, y - h * 0.62, x + w / 2 - 3 * s, y - h * 0.42], fill=(255, 255, 255))
    d.rectangle([x - 14 * s, y - h - 26 * s, x + 14 * s, y - h], fill=color, outline=INK, width=max(2, int(4 * s)))
    d.rectangle([x - 17 * s, y - h - 40 * s, x + 17 * s, y - h - 26 * s], fill=(40, 120, 220), outline=INK, width=max(2, int(4 * s)))
    d.line([(x - w / 2 + 10 * s, y - h + 12 * s), (x - w / 2 + 10 * s, y - 14 * s)], fill=(255, 255, 255), width=max(2, int(5 * s)))


def bin_(d, x, y, color, lid=0.0, s=1.0):
    """Wheelie bin with its base centre at (x, y); lid in 0..1 (open)."""
    w, h = 170 * s, 210 * s
    d.polygon([(x - w / 2, y - h), (x + w / 2, y - h), (x + w / 2 - 14 * s, y), (x - w / 2 + 14 * s, y)], fill=color, outline=INK)
    for k in (-1, 0, 1):
        d.line([(x + k * 40 * s, y - h + 30 * s), (x + k * 36 * s, y - 25 * s)], fill=tuple(max(0, c - 40) for c in color), width=int(6 * s))
    a = lid * 1.2
    px, py = x + w / 2 + 6 * s, y - h
    pts = [(px, py), (px - (w + 12 * s) * math.cos(a), py - (w + 12 * s) * math.sin(a))]
    d.line(pts, fill=tuple(max(0, c - 60) for c in color), width=int(18 * s))


def cow(d, x, y, s=1.0):
    d.rounded_rectangle([x - 90 * s, y - 120 * s, x + 70 * s, y - 50 * s], radius=int(25 * s), fill=(255, 255, 255), outline=INK, width=4)
    for sx, sy in ((-40, -100), (20, -80), (-60, -70)):
        d.ellipse([x + sx * s, y + sy * s, x + (sx + 30) * s, y + (sy + 20) * s], fill=INK)
    for lx in (-70, -40, 30, 55):
        d.line([(x + lx * s, y - 55 * s), (x + lx * s, y)], fill=INK, width=6)
    d.ellipse([x + 60 * s, y - 150 * s, x + 120 * s, y - 95 * s], fill=(255, 255, 255), outline=INK, width=4)
    d.ellipse([x + 92 * s, y - 112 * s, x + 128 * s, y - 92 * s], fill=(245, 180, 180), outline=INK, width=3)
    d.ellipse([x + 80 * s, y - 135 * s, x + 88 * s, y - 127 * s], fill=INK)
    # a plastic bag in its mouth
    d.polygon([(x + 115 * s, y - 100 * s), (x + 150 * s, y - 120 * s), (x + 160 * s, y - 80 * s)], fill=(235, 240, 245), outline=INK)


def fish(d, x, y, s=1.0, t=0.0):
    sw = math.sin(t * 8) * 8 * s
    d.ellipse([x - 70 * s, y - 35 * s, x + 50 * s, y + 35 * s], fill=(255, 170, 80), outline=INK, width=4)
    d.polygon([(x + 45 * s, y), (x + 95 * s, y - 35 * s + sw), (x + 95 * s, y + 35 * s + sw)], fill=(255, 140, 60), outline=INK)
    d.ellipse([x - 48 * s, y - 12 * s, x - 34 * s, y + 2 * s], fill=INK)
    for k in range(4):  # microplastic bits going in
        bx = x - 110 * s - k * 22 * s + (t * 40 % 22)
        d.rectangle([bx, y + (k % 2) * 8 * s, bx + 8 * s, y + 8 * s + (k % 2) * 8 * s], fill=(80, 160, 230))


def big_number(d, txt, cx, cy, size=120, color=INK, maxw=None):
    f = font(size); w = d.textlength(txt, font=f)
    maxw = maxw or W - 60
    while w > maxw and size > 20:                 # shrink very long labels to fit the frame
        size -= 2; f = font(size); w = d.textlength(txt, font=f)
    x = min(max(cx - w / 2, 30), W - 30 - w)        # keep the whole text on screen
    d.text((x, cy - size * 0.6), txt, font=f, fill=color)


def card(d, x, y, w, h, fill=(255, 255, 255)):
    d.rounded_rectangle([x, y, x + w, y + h], radius=22, fill=fill, outline=(220, 220, 215), width=3)


# ------------------------------------------------------------------ scenes (t = local time, D = scene duration)
def sc_intro(d, t, D):
    d.rectangle([0, 560, W, H], fill=(236, 214, 160))
    x = 200 + 380 * ease(t / 1.3)
    head = stickman(d, x, 600, 1.25, t, pose="wave" if t > 1.2 else "talk")
    # the bottle drops and bonks him
    if t < D * 0.55:
        by = -80 + (head[1] - 60) * ease((t - 0.8) / (D * 0.45))
        bottle(d, x + 10, by, 0.7)
    else:
        k = t - D * 0.55
        bottle(d, x + 160 + 60 * k, 600 - max(0, 40 - 260 * k), 0.7)
        for i in range(5):  # cartoon stars
            a = t * 5 + i * 1.256
            sx, sy = head[0] + 70 * math.cos(a), head[1] - 50 + 18 * math.sin(a)
            d.regular_polygon((sx, sy, 10), 5, fill=(255, 196, 0))
    # speech bubble with a wink: '?!'
    if t > 1.3:
        bx = x + 140
        d.rounded_rectangle([bx, 150, bx + 230, 280], radius=40, fill=(255, 255, 255), outline=INK, width=5)
        d.polygon([(bx + 40, 270), (bx + 10, 330), (bx + 90, 275)], fill=(255, 255, 255), outline=INK)
        big_number(d, "?!", bx + 115, 215, 90, (236, 90, 79))


def sc_harm(d, t, D):
    half = D * 0.5
    if t < half:  # bottle sinking into soil for 450 years
        d.rectangle([0, 420, W, H], fill=(140, 98, 60))
        for i in range(6):
            d.ellipse([80 + i * 200, 470 + (i % 2) * 90, 120 + i * 200, 490 + (i % 2) * 90], fill=(110, 76, 46))
        sink = 120 * ease(t / half)
        bottle(d, 420, 430 + sink, 1.4)
        big_number(d, str(int(450 * ease(t / (half * 0.85)))), 900, 230, 170)
        d.arc([780, 330, 1020, 380], 0, 180, fill=INK, width=6)  # hourglass-ish base line
        stickman(d, 150, 640, 0.8, t, pose="talk")
    else:  # India's yearly pile grows
        k = ease((t - half) / (half * 0.8))
        d.rectangle([0, 560, W, H], fill=(214, 220, 226))
        n = int(66 * k)                       # pyramid of bottles, filled from the bottom row up
        rows = 11
        for i in range(n):
            r = rows - 1 - int((math.sqrt(8 * i + 1) - 1) / 2) % rows   # bottom row first
            row_len = r + 1
            col = i % row_len
            bx = 620 + (col - r / 2) * 44
            bottle(d, bx, 230 + r * 34, 0.3, color=(150 + (i * 13) % 80, 205, 240))
        big_number(d, f"{3.4 * k:.1f}", 1020, 200, 150)
        big_number(d, "M t / yr", 1020, 310, 48, (90, 96, 105))
        stickman(d, 150, 640, 0.8, t, pose="sad" if k > 0.6 else "talk")


def sc_who(d, t, D):
    d.rectangle([0, 0, W, H], fill=(246, 241, 232))
    slots = [(70, 120), (370, 120), (670, 120), (970, 120)]
    for i, (x, y) in enumerate(slots):
        ti = t - i * (D * 0.2)
        if ti < 0:
            continue
        pop = ease(ti / 0.35)
        cw, ch = 240 * pop, 300 * pop
        card(d, x + 120 - cw / 2, y + 150 - ch / 2, cw, ch)
        if pop < 0.95:
            continue
        cx, cy = x + 120, y + 250
        if i == 0:
            cow(d, cx - 30, cy, 0.85)
        elif i == 1:
            fish(d, cx, cy - 60, 0.95, t)
        elif i == 2:
            stickman(d, cx, cy + 20, 0.62, t, pose="sad", talking=False, vest=(255, 140, 0))
        else:
            stickman(d, cx, cy + 20, 0.45, t, pose="sad", talking=False)
        r = 22 + 6 * math.sin(t * 6)
        d.ellipse([x + 205 - r, y + 15 - r, x + 205 + r, y + 15 + r], fill=(236, 90, 79))
        big_number(d, "!", x + 205, y + 18, 36, (255, 255, 255))
    stickman(d, 640, 700, 0.7, t, pose="talk")


def sc_dispose(d, t, D):
    d.rectangle([0, 560, W, H], fill=(205, 222, 235))
    bin_(d, 860, 600, BLUE_BIN, lid=ease((t - D * 0.45) / 0.4) * (1 - ease((t - D * 0.7) / 0.3)))
    p = t / D
    if p < 0.25:     # rinse
        bottle(d, 330, 560, 1.2)
        for i in range(8):
            yy = (t * 400 + i * 60) % 260 + 220
            d.ellipse([300 + (i % 3) * 25, yy, 312 + (i % 3) * 25, yy + 18], fill=(80, 160, 230))
    elif p < 0.42:   # squash
        bottle(d, 330, 560, 1.2, squash=ease((p - 0.25) / 0.12))
        d.rounded_rectangle([250, 300 + 120 * ease((p - 0.25) / 0.12), 410, 330 + 120 * ease((p - 0.25) / 0.12)], radius=10, fill=(120, 120, 120))
    elif p < 0.6:    # fly into the blue bin
        k = ease((p - 0.42) / 0.18)
        bx = 330 + (860 - 330) * k; by = 560 - 380 * math.sin(math.pi * k) + 80 * k
        bottle(d, bx, by, 1.2 * (1 - 0.4 * k), squash=1)
    # 9% recycled pie
    if p > 0.62:
        k = ease((p - 0.62) / 0.25)
        cx, cy, r = 1110, 250, 110
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 220, 215))
        d.pieslice([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * 0.09 * k, fill=GREEN_BIN)
        big_number(d, f"{int(9 * k)}%", cx, cy + 170, 80, GREEN_BIN)
    stickman(d, 130, 640, 0.8, t, pose="talk")


def sc_prevent(d, t, D):
    d.rectangle([0, 0, W, H], fill=(232, 245, 236))
    p = t / D
    # plastic bag -> cloth bag
    k = ease(p / 0.25)
    if k < 1:
        d.polygon([(240, 380), (360, 380), (390, 560), (210, 560)], fill=(240, 244, 248), outline=INK)
        d.line([(200, 360), (400, 580)], fill=RED_BIN, width=int(14 * k))
        d.line([(400, 360), (200, 580)], fill=RED_BIN, width=int(14 * k))
    else:
        d.rounded_rectangle([210, 400, 390, 570], radius=18, fill=(214, 186, 140), outline=INK, width=5)
        d.arc([250, 330, 350, 450], 180, 360, fill=INK, width=8)
    # steel bottle with a tick
    if p > 0.25:
        d.rounded_rectangle([470, 380, 540, 570], radius=20, fill=(185, 192, 200), outline=INK, width=5)
        d.rectangle([482, 350, 528, 382], fill=(120, 128, 136), outline=INK, width=4)
        d.line([(560, 470), (585, 500), (630, 430)], fill=GREEN_BIN, width=12)
    # 2050 scale: fish vs plastic
    if p > 0.45:
        k = ease((p - 0.45) / 0.25); tilt = 40 * k
        cx = 960
        d.line([(cx, 600), (cx, 380)], fill=INK, width=8)
        L = (cx - 200, 380 - tilt); R = (cx + 200, 380 + tilt)
        d.line([L, R], fill=INK, width=8)
        fish(d, L[0], L[1] - 50, 0.6, t)
        for i in range(3):
            bottle(d, R[0] - 40 + i * 40, R[1] - 10, 0.35)
        big_number(d, "2050", cx, 240, 90, (236, 90, 79))
    stickman(d, 700, 660, 0.75, t, pose="thumbs" if p > 0.8 else "talk")


SCENES = {"intro": sc_intro, "harm": sc_harm, "who": sc_who, "dispose": sc_dispose, "prevent": sc_prevent}
BG = {"intro": (255, 236, 179), "harm": (200, 228, 245), "who": (246, 241, 232), "dispose": (232, 240, 248), "prevent": (232, 245, 236)}


def main(cls="plastic_waste"):
    from build import tts_voices, synth
    out = os.path.join(HERE, cls); os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    voices = tts_voices()
    seg = {}   # (beat, lang) -> (wav path, seconds)
    for beat in ORDER:
        for lang in ("en", "hi", "ta"):
            p = os.path.join(out, "raw", f"x_{beat}_{lang}.wav")
            seg[beat, lang] = (p, synth(voices[lang], SCRIPTS[cls][beat][lang], p, lang))
    # each scene lasts as long as its longest language + a breath
    dur = {b: max(seg[b, l][1] for l in ("en", "hi", "ta")) + 0.6 for b in ORDER}
    starts, acc = {}, 0.3
    for b in ORDER:
        starts[b] = acc; acc += dur[b]
    total = acc + 0.4
    print("scenes:", {b: round(dur[b], 1) for b in ORDER}, "total", round(total, 1))
    # voice-over per language: each beat placed at its scene start
    for lang in ("en", "hi", "ta"):
        sr = None; buf = None
        for b in ORDER:
            with wave.open(seg[b, lang][0]) as w:
                sr = sr or w.getframerate()
                y = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if buf is None:
                buf = np.zeros(int(total * sr), np.int16)
            i = int(starts[b] * sr); buf[i:i + len(y)] = y[:len(buf) - i]
        with wave.open(os.path.join(out, f"explainer_{lang}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(buf.tobytes())
    # frames -> ffmpeg
    mp4 = os.path.join(out, "explainer.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    poster = None
    for f in range(int(total * FPS)):
        t = f / FPS
        beat = max((b for b in ORDER if starts[b] <= t), key=lambda b: starts[b], default="intro")
        lt = t - starts[beat]
        img = Image.new("RGB", (W, H), BG[beat]); d = ImageDraw.Draw(img)
        SCENES[beat](d, lt, dur[beat])
        # scene progress dots
        for i, b in enumerate(ORDER):
            c = (243, 189, 63) if b == beat else (200, 200, 195)
            d.ellipse([W / 2 - 70 + i * 32, H - 34, W / 2 - 56 + i * 32, H - 20], fill=c)
        # fade between scenes
        fade = min(1.0, lt / 0.25, (dur[beat] - lt) / 0.25 if beat == ORDER[-1] else 1.0)
        arr = np.asarray(img)
        if fade < 1:
            arr = (arr.astype(np.float32) * fade + 255 * (1 - fade)).astype(np.uint8)
        ff.stdin.write(arr.tobytes())
        if beat == "harm" and poster is None and lt > 2:
            poster = Image.fromarray(arr); poster.save(os.path.join(out, "explainer.jpg"), quality=85)
    ff.stdin.close(); ff.wait()
    # register as the first clip for this class
    idx_p = os.path.join(HERE, "index.json")
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    vo = {l: " ".join(SCRIPTS[cls][b][l] for b in ORDER) for l in ("en", "hi", "ta")}
    entry = dict(id="explainer", title={"en": "Plastic: the 40-second story", "hi": "प्लास्टिक: 40 सेकंड की कहानी",
                                        "ta": "பிளாஸ்டிக்: 40 செகண்ட் கதை"},
                 video=f"{cls}/explainer.mp4", vo_files={l: f"{cls}/explainer_{l}.wav" for l in ("en", "hi", "ta")},
                 poster=f"{cls}/explainer.jpg", credit="Sorti (drawn in code)", page="", vo=vo)
    idx[cls] = [entry] + [c for c in idx.get(cls, []) if c["id"] != "explainer"]
    json.dump(idx, open(idx_p, "w"), indent=1, ensure_ascii=False)
    print("done:", mp4, f"{total:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plastic_waste")
