#!/usr/bin/env python3
"""Review contact sheet for the current, unmodified PIL stickman."""
from PIL import Image, ImageDraw, ImageFont
from explainer import stickman


def main():
    cw, ch = 320, 410
    image = Image.new("RGB", (cw * 4, ch * 7), (246, 241, 232))
    d = ImageDraw.Draw(image)
    label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)

    def cell(row, col, pose, scale, time=0.0, talking=True, vest=None):
        left, top = col * cw, row * ch
        d.rectangle((left, top, left + cw - 1, top + ch - 1), outline=(200, 200, 195))
        caption = f"{pose} s={scale:g} t={time:.3f}"
        d.text((left + 12, top + 10), caption, font=label, fill=(34, 40, 49))
        d.text((left + 12, top + 32), f"talking={talking}" + (" vest" if vest else ""), font=label, fill=(34, 40, 49))
        d.line((left + 35, top + 380, left + cw - 35, top + 380), fill=(210, 205, 195), width=1)
        stickman(d, left + cw / 2, top + 380, scale, time, pose, talking, vest)

    poses = ("talk", "wave", "sad", "thumbs")
    for row, scale in enumerate((1.25, 0.8, 0.5)):
        for col, pose in enumerate(poses):
            cell(row, col, pose, scale)
    for col, (scale, vest) in enumerate(((0.62, (255, 140, 0)), (0.45, None), (0.6, None), (0.5, (255, 140, 0)))):
        cell(3, col, "sad", scale, talking=False, vest=vest)
    for row, pose in enumerate(("talk", "wave", "sad"), start=4):
        for col, time in enumerate((0.0, 0.262, 0.524, 0.785)):
            cell(row, col, pose, 0.8, time)
    image.save("/tmp/stick_preview.jpg", quality=95)
    print("/tmp/stick_preview.jpg")


if __name__ == "__main__":
    main()
