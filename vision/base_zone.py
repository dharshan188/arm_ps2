"""The cardboard base: only items sitting ON it are detected; the board itself, things below it, beside it or far
above it (monitor, people) are ignored.

The board is found by its tan colour (HSV), as the largest wide tan region, re-checked every `every` frames.
The last good position is kept while items cover it, and saved to disk so a restart knows it immediately.
"""
import json
import os

import cv2
import numpy as np


class BaseZone:
    def __init__(self, path=None, every=15, lo=(15, 55, 40), hi=(28, 210, 235), min_area=.012):
        self.path, self.every, self.lo, self.hi, self.min_area = path, every, lo, hi, min_area
        self.box, self.n, self.cand, self.cand_n = None, 0, None, 0
        self._ii = self._ii_src = None
        env = os.environ.get("SORTI_BASE_BOX")
        if env:
            self.box = tuple(int(v) for v in env.split(","))
        elif path and os.path.exists(path):
            try:
                self.box = tuple(json.load(open(path))["box"])
            except (OSError, ValueError, KeyError):
                self.box = None

    def find(self, bgr):
        """Tan board bounding box (x0, y0, x1, y1) in this frame, or None."""
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (w // 2, h // 2))
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (7, 7), 0), cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lo, self.hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        best = None
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c) * 4 / (w * h)
            cy = (y + bh / 2) * 2
            if (area >= self.min_area and bw >= 1.2 * bh and cy >= .4 * h      # wide, in the lower part: a board, not a face
                    and (best is None or area > best[0])):
                best = (area, (2 * x, 2 * y, 2 * (x + bw), 2 * (y + bh)))
        return best[1] if best else None

    def update(self, bgr):
        """Constant base: once known (saved file, SORTI_BASE_BOX, or the first sighting) it never changes.
        Delete pi/base_zone.json to recalibrate after moving the camera or the board."""
        if self.box is not None:
            return self.box
        self.n += 1
        if self.n % self.every != 1:
            return None
        found = self.find(bgr)
        if found is not None:
            self.box = tuple(int(v) for v in found)
            if self.path:
                try:
                    json.dump({"box": list(self.box)}, open(self.path, "w"))
                except OSError:
                    pass
        return self.box

    def mask(self, bgr):
        """Pixels of the cardboard board itself (the largest tan region), slightly grown, at full resolution.
        Items lying on it keep their own colours, so they are not in the mask. None if no board is visible."""
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (w // 2, h // 2))
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lo, self.hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] * 4 < self.min_area * w * h:
            return None
        board = (lab == k).astype(np.uint8) * 255
        board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # fill corrugation gaps
        board = cv2.dilate(board, np.ones((5, 5), np.uint8))
        x, y, bw, bh = cv2.boundingRect(board)
        self.box = (2 * x, 2 * y, 2 * (x + bw), 2 * (y + bh))
        return cv2.resize(board, (w, h), interpolation=cv2.INTER_NEAREST) > 0

    def on_board(self, box, board, h):
        """Detection box kept if it is not mostly cardboard and sits over the board (not beside / high above it).
        board = the current mask() (None: no board visible, keep everything)."""
        if board is None or self.box is None:
            return True
        if self._ii is None or self._ii_src is not board:
            self._ii, self._ii_src = cv2.integral(board.astype(np.uint8)), board
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(board.shape[1], x1), min(board.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return False
        ii = self._ii
        frac = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((x1 - x0) * (y1 - y0))
        if frac >= .3:                                                    # mostly cardboard: the base itself
            return False
        bx0, by0, bx1, by1 = self.box
        bw = bx1 - bx0
        cx = (x0 + x1) / 2
        return bx0 - .1 * bw <= cx <= bx1 + .1 * bw and y1 >= by0 - .05 * h and y0 <= by1

    def keep(self, box):
        """True if a detection box sits on the board (or if no board is known: no filtering)."""
        if self.box is None:
            return True
        bx0, by0, bx1, by1 = self.box
        bw, bh = bx1 - bx0, by1 - by0
        x0, y0, x1, y1 = box[:4]
        cx = (x0 + x1) / 2
        if not (bx0 - .15 * bw <= cx <= bx1 + .15 * bw):                 # beside the board
            return False
        if not (by0 - .5 * bh <= y1 <= by0 + .5 * bh):                    # bottom edge not on the board's top surface
            return False
        ix = max(0, min(x1, bx1) - max(x0, bx0)) * max(0, min(y1, by1) - max(y0, by0))
        if ix >= .5 * bw * bh:                                            # the board itself
            return False
        return True

    def draw(self, frame):
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            cv2.line(frame, (x0, y0), (x1, y0), (184, 212, 0), 2)         # the surface items must sit on (teal)
            cv2.putText(frame, "BASE", (x0 + 6, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (184, 212, 0), 1, cv2.LINE_AA)
