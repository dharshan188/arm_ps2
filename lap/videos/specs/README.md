# Explainer specs (one JSON per video, rendered by ../render.py)

File name: specs/<cls>__<slug>.json. One video is about 40 s, with 5 beats (each about 6–10 s of speech, 15–25 English words):
1. **intro**: a comic hook from the stick-man narrator (the item falls on his head). Address the viewer, be playful, mention the item.
2. **harm**: why it's bad, with 1–2 REAL statistics (shown as big animated numbers).
3. **who**: who it affects (2–4 victims drawn as cards).
4. **dispose**: how to dispose of it (prep steps, then the item flies into the right bin), plus an optional stat.
5. **prevent**: how to avoid or reduce it (the item is crossed out and swapped for a better one), plus an optional stat. End with a fun line.

```json
{"cls": "plastic_waste", "slug": "bags", "item": "bag",
 "title": {"en": "...", "hi": "...", "ta": "..."},
 "beats": {
  "intro":   {"en": "...", "hi": "...", "ta": "..."},
  "harm":    {"en": "...", "hi": "...", "ta": "...",
              "stats": [{"value": 450, "unit": "years"}, {"value": 3.4, "decimals": 1, "unit": "million t / yr"}]},
  "who":     {"en": "...", "hi": "...", "ta": "...", "victims": ["cow", "fish", "worker"]},
  "dispose": {"en": "...", "hi": "...", "ta": "...", "bin": "blue", "steps": ["rinse", "squash"],
              "stat": {"value": 9, "suffix": "%"}},
  "prevent": {"en": "...", "hi": "...", "ta": "...", "swap_from": "bag", "swap_to": "cloth_bag",
              "swap_to2": "steel_bottle", "stat": {"value": 2050, "bad": true}}
 },
 "sources": ["CPCB Annual Report 2019-20 (plastic waste 3.47 MT)", "..."]}
```
Fields:
- **bin**: blue (dry, recyclable), green (wet/food), red (hazardous: batteries, CFL/tubes), ewaste (e-waste collection centre).
- **steps**: any of rinse, squash, tape, dry, wrap, box (0–2 of them).
- **stat**: value (number), plus optional decimals, prefix, suffix ("%"), unit (short English label under the number, e.g. "years", "million t / yr", "kg / person"), and bad (red colour).
- **item, swap_from, swap_to, swap_to2**: must be keys from ITEMS. **victims**: keys from VICTIMS. Any other key falls back to a bottle, so use only these:
ITEMS: apple, bag, battery, bottle, box, broken_glass, bulb, button_cell, cable, can, cfl, charger, cloth_bag, compost_bin, container, cup, donate_box, earphones, flowers, foil, food, glass_bottle, jar, jeans, laptop, leaves, led_bulb, milk_packet, newspaper, notebook, paper_bag, paper_cup, peel, phone, rechargeable, remote, saree, shirt, steel_bottle, steel_cup, straw, tea, tiffin, tin, tube, wrapper, wrench
VICTIMS: air, bird, child, cow, dog, farmer, fish, person, river, soil, worker

Language rules (same as the app):
- **Tamil:** casual spoken Chennai Tamil with English words in Tamil script (ரீசைக்கிள், பின், பிளாஸ்டிக்). No Latin letters, no literary Tamil.
- **Hindi:** Hinglish in Devanagari (डालो, रीसायकल, बिन).
- **Numbers:** write them as words in all three languages, because the TTS reads digits badly ("four hundred and fifty" / "साढ़े चार सौ" / "நானூத்தி அம்பது").
- **Statistics:** REAL and conservative. Name each source in "sources". Prefer Indian sources (CPCB, MoEFCC, NITI Aayog, UNEP India) and global ones (OECD, UNEP, WHO, Global E-waste Monitor 2024, FAO). If unsure, round down or say "about" / "lakhs of".
- **Tone:** a funny, friendly stick-man narrator who talks to the viewer ("Hey you!", "Deal? High five!"). Never preachy.
