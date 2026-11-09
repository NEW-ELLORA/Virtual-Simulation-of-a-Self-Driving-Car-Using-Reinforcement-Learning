"""
City Driving Simulator (pygame)  -  random cities, flyovers, missions
=====================================================================
A drivable "GTA-lite" city for manual driving, dataset recording and CNN autopilot.

WORLD  (a NEW random city every run; --seed N reproduces one)
  * Road types: HIGHWAY (dual carriageway, median barrier, guard rails, 80 km/h),
    AVENUE (2 lanes each way, planted median, 60), STREET (40), NARROW LANE (25),
    ONE-WAY street (2 lanes, arrows, 40).
  * Winding / curved roads, T-junctions, corners, removed streets, ring road.
  * FLYOVERS: real elevated bridges with ramps, parapets, lamps and pillars.
    Drive over them or under them (different levels never collide).
  * Working traffic signals, crosswalks (signalled + zebra), pedestrians.
  * Traffic: sedans, SUVs, trucks, BUSES, AUTO-RICKSHAWS, MOTORBIKES, SCOOTERS.
    They follow random destinations, obey signals, yield and use the flyovers.

YOU
  * Choose your vehicle (X): sedan, SUV, auto-rickshaw, motorbike, scooter.
  * Smooth pedal, speed limiter (slider), cruise control, calm speed-sensitive steering.
  * Any hit = crash -> short freeze -> automatic restart.
  * Rear-view mirror, full REAR CAMERA (B) and automatic REVERSE CAMERA.
  * Busy roads: traffic is streamed around you, so there are always vehicles ahead in
    your lane and coming towards you in the opposite lane, wherever you drive.

INSTALL:   pip install pygame numpy opencv-python      (torch only for --model)
RUN:       python driving_sim.py
           python driving_sim.py --seed 7 --time night --vehicle bike
           python driving_sim.py --traffic 200     (even more vehicles)
           python driving_sim.py --model net.pt    (CNN autopilot)
           python driving_sim.py --headless --max-frames 600

CONTROLS
    W / Up       throttle (progressive)       S / Down  brake (hold at standstill = reverse)
    A,D / Arrows steer                        C  cruise control    + / -  cruise speed
    V   view: 1st person -> chase -> top-down   B  rear camera (toggle)
    M   city map        X  change vehicle       N  next start location
    K   generate a NEW random city              T  time of day     R  restart
    SPACE  record dataset    P  autopilot       ESC  quit          mouse wheel: zoom (top-down)
    Sliders (mouse): Max Speed, Turn Sensitivity
"""
import os
import sys
import math
import time
import random
import argparse
import bisect
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pygame
from astar import AStarRouter
from rl_agent import AutoDriveController

TORCH_AVAILABLE = None
torch = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# =============================================================================
# CONFIGURATION  (all distances in METRES, times in SECONDS, speeds in m/s)
# =============================================================================
class Config:
    # --- Display ---
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 750
    FPS = 60
    PHYS_DT = 1.0 / 60.0

    # --- City layout ---
    GRID_N = 9                  # 9 x 9 grid lines
    BLOCK = 120.0
    BLOCK_MIN = 108.0           # random spacing between grid lines
    BLOCK_MAX = 150.0
    H_LINES = (0, 4, 8)         # grid lines that are highways (rows and columns)
    CONNECTORS = (2, 6)         # streets that run all the way to the highways
    STREET_HALF = 3.8           # half width of a street roadway (1 lane each way)
    HIGHWAY_HALF = 8.4          # half width of a highway roadway
    MEDIAN_HALF = 1.2           # half width of the highway median
    SIDEWALK_W = 3.2
    SHOULDER_W = 1.6
    LANES_STREET = (1.9,)       # lane centre offsets to the right of travel
    LANES_HIGHWAY = (5.9, 3.3)  # [0] = outer/right lane, [1] = inner/left lane
    LEAD = 7.5                  # lane ends this far before the junction box (gives wide turn arcs)
    BEND_D_HIGHWAY = 90.0       # corner "radius" of the ring-road corners
    BEND_D_STREET = 16.0
    LIMIT_STREET = 11.1         # 40 km/h
    LIMIT_HIGHWAY = 22.2        # 80 km/h
    MAP_MARGIN = 100.0
    WALL_OFFSET = 110.0         # city boundary wall distance outside the ring road

    # --- Player vehicle ---
    CAR_LENGTH = 4.5
    CAR_WIDTH = 1.9
    WHEELBASE = 2.7
    DEFAULT_MAX_SPEED = 16.7    # ~60 km/h : comfortable default, slider changes it
    MAX_SPEED = 16.7
    SPEED_SLIDER_MIN = 6.0
    SPEED_SLIDER_MAX = 36.0
    MAX_REVERSE_SPEED = 4.0
    ENGINE_ACCEL = 3.6          # m/s^2 at full pedal from standstill
    BRAKE_DECEL = 9.0
    ENGINE_BRAKING = 0.9        # m/s^2 when coasting
    ROLL_RESISTANCE = 0.12
    AIR_DRAG = 0.0012
    MAX_STEER_ANGLE = 0.55      # rad at standstill
    MAX_LAT_ACCEL = 7.0         # m/s^2: limits wheel angle at speed
    DEFAULT_STEER_SENS = 0.7    # CALM default (was far too twitchy)
    STEER_SENSITIVITY = 0.7
    STEER_RATE = 1.7            # input ramp per second at sensitivity 1.0
    STEER_RETURN_RATE = 2.6
    PEDAL_RISE = 3.5            # throttle ramp /s
    PEDAL_FALL = 3.5
    BRAKE_RISE = 4.5
    CRUISE_STEP = 5.0 / 3.6

    # --- Traffic / pedestrians ---
    TRAFFIC_COUNT = 65          # vehicles kept alive around the player (streamed)
    TRAFFIC_RADIUS = 160.0      # Radius method: active traffic populated only within immediate driving radius
    PED_ACTIVE_RADIUS = 110.0   # Radius method: active pedestrians only within immediate radius
    PED_MAX = 20
    PED_SPEED = (1.15, 1.55)

    # --- Camera & Radius-Based Rendering ---
    HORIZON_Y = 330.0
    FOCAL_LENGTH = 520.0
    CAM_HEIGHT = 1.28
    Z_NEAR = 0.4
    DRAW_DIST = 170.0           # Radius method: maximum draw distance
    RENDER_RADIUS = 160.0       # Radius method: active road & ground geometry radius
    RENDER_OBJ_RADIUS = 140.0   # Radius method: active 3D buildings, props & vehicles radius
    DEFAULT_VIEW = "FIRST_PERSON"   # FIRST_PERSON / CHASE / TOP_DOWN

    # --- Observation / CNN sensor ---
    OBS_WIDTH = 200
    OBS_HEIGHT = 66

    # --- Dataset ---
    DATA_DIR = "data"
    IMAGE_DIR = os.path.join("data", "images")
    LOG_FILE = os.path.join("data", "log.csv")
    RECORD_INTERVAL = 2

    # --- Rules ---
    CRASH_COOLDOWN_SEC = 1.3
    RESPAWN_SHIELD_SEC = 2.5
    RANDOM_SEED = None


# =============================================================================
# MATH / GEOMETRY HELPERS
# =============================================================================
def clamp(v, a, b):
    return a if v < a else (b if v > b else v)


def normalize_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def mix(c1, c2, t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def shade(c, k):
    return (max(0, min(255, int(c[0] * k))),
            max(0, min(255, int(c[1] * k))),
            max(0, min(255, int(c[2] * k))))


def obb_corners(cx, cy, heading, length, width):
    """4 corners (front-right, front-left, rear-left, rear-right). Heading 0 = +X."""
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length * 0.5, width * 0.5
    fx, fy = c * hl, s * hl
    rx, ry = -s * hw, c * hw
    return [(cx + fx + rx, cy + fy + ry), (cx + fx - rx, cy + fy - ry),
            (cx - fx - rx, cy - fy - ry), (cx - fx + rx, cy - fy + ry)]


def sat_polygon_collision(pa, pb):
    """Separating Axis Theorem for two convex polygons."""
    for poly in (pa, pb):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ax, ay = -(y2 - y1), (x2 - x1)
            ln = math.hypot(ax, ay)
            if ln == 0:
                continue
            ax /= ln
            ay /= ln
            mina = min(p[0] * ax + p[1] * ay for p in pa)
            maxa = max(p[0] * ax + p[1] * ay for p in pa)
            minb = min(p[0] * ax + p[1] * ay for p in pb)
            maxb = max(p[0] * ax + p[1] * ay for p in pb)
            if maxa < minb or maxb < mina:
                return False
    return True


def circle_obb_collision(cx, cy, r, ox, oy, heading, length, width):
    """Circle vs oriented box."""
    dx, dy = cx - ox, cy - oy
    c, s = math.cos(heading), math.sin(heading)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    qx = clamp(lx, -length * 0.5, length * 0.5)
    qy = clamp(ly, -width * 0.5, width * 0.5)
    return (lx - qx) ** 2 + (ly - qy) ** 2 < r * r


def seg_point_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(px - x1, py - y1), 0.0
    t = clamp(((px - x1) * dx + (py - y1) * dy) / l2, 0.0, 1.0)
    return math.hypot(px - (x1 + dx * t), py - (y1 + dy * t)), t


def bezier2(p0, p1, p2, n):
    pts = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) * (1 - t)
        b = 2 * (1 - t) * t
        c = t * t
        pts.append((a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1]))
    return pts


def bezier3(p0, p1, p2, p3, n):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        a = u * u * u
        b = 3.0 * u * u * t
        c = 3.0 * u * t * t
        d = t * t * t
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def poly_contains(poly, x, y):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def strip_polygon(pts, half_w_left, half_w_right):
    """Closed polygon around a polyline (left/right widths, y-down coordinates)."""
    left, right = [], []
    n = len(pts)
    for i in range(n):
        if i == 0:
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif i == n - 1:
            dx, dy = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln, dx / ln          # right-hand normal (y-down)
        left.append((pts[i][0] - nx * half_w_left, pts[i][1] - ny * half_w_left))
        right.append((pts[i][0] + nx * half_w_right, pts[i][1] + ny * half_w_right))
    return left + right[::-1]


class SpatialHash:
    """Tiny grid hash for static objects (stores arbitrary items by bounding box)."""

    def __init__(self, cell=50.0):
        self.cell = cell
        self.cells = defaultdict(list)

    def insert(self, item, x0, y0, x1, y1):
        c = self.cell
        for gx in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for gy in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                self.cells[(gx, gy)].append(item)

    def query_box(self, x0, y0, x1, y1):
        c = self.cell
        seen = set()
        out = []
        for gx in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for gy in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                for it in self.cells.get((gx, gy), ()):
                    k = id(it)
                    if k not in seen:
                        seen.add(k)
                        out.append(it)
        return out


# =============================================================================
# LIGHTING PALETTES
# =============================================================================
PALETTES = {
    "DAY": dict(sky_top=(46, 104, 200), sky_hor=(178, 208, 232), fog=(184, 208, 228),
                tint=(1.0, 1.0, 1.0), fog0=70.0, fog1=430.0, night=False, glow=0.0,
                sun=(255, 244, 205), cloud=(250, 252, 255), hills=(122, 146, 176),
                lit=0.0),
    "SUNSET": dict(sky_top=(52, 58, 128), sky_hor=(255, 156, 92), fog=(226, 150, 118),
                   tint=(1.0, 0.82, 0.70), fog0=50.0, fog1=380.0, night=False, glow=0.6,
                   sun=(255, 190, 110), cloud=(255, 190, 150), hills=(124, 92, 120),
                   lit=0.35),
    "NIGHT": dict(sky_top=(4, 8, 24), sky_hor=(22, 32, 62), fog=(12, 18, 38),
                  tint=(0.42, 0.50, 0.78), fog0=25.0, fog1=300.0, night=True, glow=1.0,
                  sun=(214, 224, 255), cloud=(60, 72, 110), hills=(24, 32, 56),
                  lit=0.7),
}
TIME_ORDER = ["DAY", "SUNSET", "NIGHT"]

# Vehicle body styles: len/wid multipliers, chassis height, cabin (rear frac, front frac, roof h)
BODY_SPECS = {
    "sedan": dict(len=1.00, wid=1.00, h_ch=0.95, cab=(0.26, 0.30, 1.45)),
    "suv":   dict(len=1.04, wid=1.05, h_ch=1.10, cab=(0.08, 0.28, 1.80)),
    "truck": dict(len=1.60, wid=1.14, h_ch=1.15, cab=None),
}

CAR_COLORS = [(205, 48, 42), (236, 170, 36), (38, 110, 205), (222, 226, 230),
              (36, 150, 84), (140, 70, 185), (224, 96, 30), (52, 54, 60), (150, 160, 170)]


# =============================================================================
# ROAD TYPES
# =============================================================================
class RoadType:
    def __init__(self, key, name, half, offs, median, barrier, limit_kmh, side_w, style, rank,
                 road_col, side_col, oneway=False, furniture='street'):
        self.key, self.name = key, name
        self.half = half                  # half width of the paved roadway
        self.offs = offs                  # lane centre offsets, to the RIGHT of travel (index 0 = outer lane)
        self.median = median              # half width of the central median (0 = none)
        self.barrier = barrier            # concrete median barrier + guard rails
        self.limit_kmh = limit_kmh
        self.limit = limit_kmh / 3.6
        self.side_w = side_w              # sidewalk / shoulder width
        self.style = style                # divided | avenue | street | lane | oneway
        self.rank = rank
        self.road_col = road_col
        self.side_col = side_col
        self.oneway = oneway
        self.furniture = furniture
        self.nlanes = len(offs)


RT = {
    'H': RoadType('H', 'HIGHWAY', 8.4, (5.9, 3.3), 1.2, True, 80, 1.6, 'divided', 3,
                  (54, 56, 60), (120, 118, 112), furniture='highway'),
    'A': RoadType('A', 'AVENUE', 7.0, (5.5, 2.6), 0.9, False, 60, 3.2, 'avenue', 2,
                  (60, 62, 66), (152, 150, 144), furniture='avenue'),
    'S': RoadType('S', 'STREET', 3.8, (1.9,), 0.0, False, 40, 3.2, 'street', 1,
                  (64, 66, 70), (152, 150, 144)),
    'L': RoadType('L', 'NARROW LANE', 3.0, (1.5,), 0.0, False, 25, 2.4, 'lane', 0,
                  (84, 80, 74), (146, 140, 130), furniture='lane'),
    'O': RoadType('O', 'ONE-WAY', 3.9, (1.85, -1.85), 0.0, False, 40, 3.2, 'oneway', 1,
                  (62, 64, 72), (152, 150, 144), oneway=True),
}


# =============================================================================
# VEHICLE CATALOGUE  (player + AI)
# =============================================================================
VEHICLES = {
    'sedan':   dict(label='Sedan', length=4.5, width=1.90, wheelbase=2.70, height=1.45, kind='car',
                    max_steer=0.55, accel=3.6, brake=9.0, vmax=42.0, eye=1.28, eye_fwd=0.35,
                    ai_w=0.24, pace=(0.82, 1.0), a_max=2.2, b=3.0, s0=2.4),
    'suv':     dict(label='SUV', length=4.8, width=2.00, wheelbase=2.90, height=1.80, kind='car',
                    max_steer=0.52, accel=3.3, brake=8.5, vmax=40.0, eye=1.55, eye_fwd=0.35,
                    ai_w=0.12, pace=(0.82, 1.0), a_max=2.0, b=3.0, s0=2.4),
    'truck':   dict(label='Truck', length=7.4, width=2.30, wheelbase=4.3, height=3.0, kind='truck',
                    max_steer=0.50, accel=2.0, brake=7.0, vmax=26.0, eye=2.3, eye_fwd=2.0,
                    ai_w=0.05, pace=(0.72, 0.9), a_max=1.2, b=2.6, s0=3.0),
    'bus':     dict(label='Bus', length=11.0, width=2.55, wheelbase=5.6, height=3.2, kind='bus',
                    max_steer=0.45, accel=1.8, brake=6.5, vmax=24.0, eye=2.5, eye_fwd=4.0,
                    ai_w=0.06, pace=(0.70, 0.88), a_max=1.1, b=2.4, s0=3.2),
    'auto':    dict(label='Auto-rickshaw', length=2.75, width=1.40, wheelbase=2.0, height=1.75, kind='auto',
                    max_steer=0.78, accel=2.4, brake=6.5, vmax=17.5, eye=1.35, eye_fwd=0.55,
                    ai_w=0.15, pace=(0.70, 0.9), a_max=1.6, b=3.0, s0=2.0),
    'bike':    dict(label='Motorbike', length=2.05, width=0.78, wheelbase=1.35, height=1.5, kind='bike',
                    max_steer=0.62, accel=4.8, brake=8.0, vmax=40.0, eye=1.42, eye_fwd=0.15,
                    ai_w=0.20, pace=(0.85, 1.0), a_max=3.0, b=3.4, s0=1.6),
    'scooter': dict(label='Scooter', length=1.85, width=0.72, wheelbase=1.25, height=1.45, kind='scooter',
                    max_steer=0.66, accel=2.8, brake=6.5, vmax=22.0, eye=1.38, eye_fwd=0.10,
                    ai_w=0.18, pace=(0.72, 0.9), a_max=2.0, b=3.0, s0=1.6),
}
PLAYER_VEHICLES = ['sedan', 'suv', 'auto', 'bike', 'scooter']

BUS_COLORS = [((196, 40, 40), (240, 240, 240)), ((30, 90, 170), (240, 240, 240)),
              ((236, 190, 30), (30, 30, 34)), ((36, 140, 80), (240, 240, 240))]
AUTO_COLORS = [((240, 200, 20), (30, 30, 34)), ((240, 200, 20), (30, 120, 60)), ((30, 30, 34), (240, 200, 20))]
RIDER_COLORS = [(210, 60, 50), (40, 90, 190), (50, 50, 56), (230, 230, 230), (40, 140, 80), (230, 150, 30)]

BUILDING_COLORS = [(150, 142, 128), (172, 160, 140), (120, 128, 142), (138, 112, 98),
                   (96, 108, 126), (180, 176, 170), (140, 155, 175), (160, 120, 100),
                   (110, 120, 112)]


# =============================================================================
# DIRECTIONS (world: x = east, y = south / screen-down; heading 0 = +x)
# =============================================================================
U = {'E': (1.0, 0.0), 'S': (0.0, 1.0), 'W': (-1.0, 0.0), 'N': (0.0, -1.0)}
HDG = {'E': 0.0, 'S': math.pi / 2, 'W': math.pi, 'N': -math.pi / 2}
OPP = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}
RIGHT = {'E': 'S', 'S': 'W', 'W': 'N', 'N': 'E'}
LEFT = {v: k for k, v in RIGHT.items()}


def right_vec(d):
    ux, uy = U[d]
    return (-uy, ux)


def axis_of(d):
    return 'EW' if d in ('E', 'W') else 'NS'


# =============================================================================
# UI SLIDER
# =============================================================================
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, format_str="{:.2f}"):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.val = float(initial_val)
        self.label = label
        self.format_str = format_str
        self.dragging = False
        self.handle_width = 14
        self.hovered = False

    def get_handle_rect(self):
        t = (self.val - self.min_val) / max(1e-5, (self.max_val - self.min_val))
        hx = self.rect.x + int(t * (self.rect.width - self.handle_width))
        return pygame.Rect(hx, self.rect.y - 4, self.handle_width, self.rect.height + 8)

    def handle_event(self, event):
        hit = self.rect.inflate(26, 26)
        hit.top -= 14
        if event.type == pygame.MOUSEMOTION:
            self.hovered = hit.collidepoint(event.pos)
            if self.dragging:
                self._update(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hit.collidepoint(event.pos):
                self.dragging = True
                self._update(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        return False

    def _update(self, mx):
        span = float(self.rect.width - self.handle_width)
        t = clamp((mx - (self.rect.x + self.handle_width // 2)) / max(1.0, span), 0.0, 1.0)
        self.val = self.min_val + t * (self.max_val - self.min_val)

    def draw(self, surface, font):
        bg = (28, 34, 46) if not self.hovered else (38, 48, 64)
        border = (0, 210, 255) if (self.dragging or self.hovered) else (74, 90, 112)
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, 1, border_radius=6)
        h = self.get_handle_rect()
        aw = max(0, h.centerx - self.rect.x)
        if aw > 0:
            pygame.draw.rect(surface, (0, 190, 240), (self.rect.x, self.rect.y, aw, self.rect.height), border_radius=6)
        pygame.draw.rect(surface, (255, 255, 255), h, border_radius=4)
        pygame.draw.rect(surface, (0, 220, 255) if (self.dragging or self.hovered) else (120, 140, 160), h, 2, border_radius=4)
        txt = font.render(f"{self.label}: {self.format_str.format(self.val)}", True, (225, 232, 245))
        surface.blit(txt, (self.rect.x, self.rect.y - 17))


# =============================================================================
# TRAFFIC SIGNAL  (2 phases: East-West / North-South, yellow + all-red clearance)
# =============================================================================
class Signal:
    YELLOW = 2.5
    ALL_RED = 1.5

    def __init__(self, green_ew, green_ns, offset):
        self.g = {'EW': green_ew, 'NS': green_ns}
        self.offset = offset
        self.segs = [('EW', 'G', green_ew), ('EW', 'Y', self.YELLOW), (None, 'R', self.ALL_RED),
                     ('NS', 'G', green_ns), ('NS', 'Y', self.YELLOW), (None, 'R', self.ALL_RED)]
        self.cycle = sum(s[2] for s in self.segs)

    def _seg(self, t):
        x = (t + self.offset) % self.cycle
        for ax, st, dur in self.segs:
            if x < dur:
                return ax, st, dur - x
            x -= dur
        return None, 'R', 0.0

    def state(self, axis, t):
        """('G'|'Y'|'R') for traffic travelling along `axis` ('EW' or 'NS')."""
        ax, st, _ = self._seg(t)
        return st if ax == axis else 'R'

    def green_left(self, axis, t):
        """Seconds of green left for `axis` (0 when not green)."""
        ax, st, rem = self._seg(t)
        return rem if (ax == axis and st == 'G') else 0.0


# =============================================================================
# POLYLINE HELPERS  (roads can be curved, so everything works on polylines)
# =============================================================================
def pl_cum(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return cum


def pl_point(pts, cum, s):
    """(x, y, heading) at arclength s."""
    n = len(pts)
    if s <= 0.0:
        i, t = 0, 0.0
    elif s >= cum[-1]:
        i, t = n - 2, 1.0
    else:
        i = max(0, min(n - 2, bisect.bisect_right(cum, s) - 1))
        seg = cum[i + 1] - cum[i]
        t = (s - cum[i]) / seg if seg > 1e-9 else 0.0
    a, b = pts[i], pts[i + 1]
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, math.atan2(b[1] - a[1], b[0] - a[0]))


def pl_slice(pts, cum, s0, s1):
    s0 = max(0.0, s0)
    s1 = min(cum[-1], s1)
    if s1 <= s0 + 0.02:
        s1 = min(cum[-1], s0 + 0.05)
        s0 = max(0.0, s1 - 0.05)
    out = [pl_point(pts, cum, s0)[:2]]
    for i in range(len(pts)):
        if s0 + 1e-6 < cum[i] < s1 - 1e-6:
            out.append(pts[i])
    out.append(pl_point(pts, cum, s1)[:2])
    return out


def pl_offset(pts, o):
    """Offset polyline by o to the RIGHT of its direction of travel (y-down world)."""
    n = len(pts)
    out = []
    for i in range(n):
        if i == 0:
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif i == n - 1:
            dx, dy = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        ln = math.hypot(dx, dy) or 1.0
        out.append((pts[i][0] - dy / ln * o, pts[i][1] + dx / ln * o))
    return out


def pl_strip(pts, o0, o1):
    return pl_offset(pts, o0) + pl_offset(pts, o1)[::-1]


def pl_min_radius(pts):
    rmin = 1e9
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ca = math.hypot(a[0] - c[0], a[1] - c[1])
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 < 1e-9:
            continue
        rmin = min(rmin, ab * bc * ca / (2.0 * area2))
    return rmin


class _Regen(Exception):
    pass


class Elev:
    """Elevation profile of a flyover edge: ramp up, plateau, ramp down (smooth)."""

    def __init__(self, height, ramp, length):
        self.H, self.R, self.L = height, ramp, length

    def z(self, u):
        d = min(u, self.L - u)
        t = 0.0 if d <= 0 else (1.0 if d >= self.R else d / self.R)
        return self.H * t * t * (3.0 - 2.0 * t)


# =============================================================================
# ROAD NETWORK PRIMITIVES
# =============================================================================
class Path:
    """A drivable polyline: either a LANE (between junctions) or a CONNECTOR (through a junction)."""
    _ids = 0

    def __init__(self, pts, kind, limit):
        Path._ids += 1
        self.pid = Path._ids
        self.pts = pts
        self.kind = kind                # 'lane' | 'conn'
        self.limit = limit
        self.cap = limit
        cum = [0.0]
        hd = []
        for i in range(1, len(pts)):
            dx, dy = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
            cum.append(cum[-1] + math.hypot(dx, dy))
            hd.append(math.atan2(dy, dx))
        self.cum = cum
        self.length = cum[-1]
        self.hd = hd
        vh = [hd[0]]
        for i in range(1, len(hd)):
            vh.append(math.atan2(math.sin(hd[i - 1]) + math.sin(hd[i]), math.cos(hd[i - 1]) + math.cos(hd[i])))
        vh.append(hd[-1])
        self.vh = vh
        self.out = []
        self.stop_s = None
        self.node = None
        self.signal = None
        self.axis = None
        self.move = 'S'
        self.opp = []
        self.edge = None
        self.tdir = None
        self.idx = 0
        self.in_lane = None
        self.out_lane = None
        self.conflicts = set()
        self.rmin = 1e9
        self.elev = None
        self.u0 = 0.0
        self.du = 1.0

    def zat(self, s):
        if self.elev is None:
            return 0.0
        return self.elev.z(self.u0 + self.du * s)

    def pose(self, s):
        pts = self.pts
        if len(pts) == 2:
            L = self.length
            t = clamp(s / L, 0.0, 1.0) if L > 1e-6 else 0.0
            return (pts[0][0] + (pts[1][0] - pts[0][0]) * t,
                    pts[0][1] + (pts[1][1] - pts[0][1]) * t, self.hd[0])
        cum = self.cum
        n = len(pts)
        if s <= 0.0:
            i, t = 0, 0.0
        elif s >= self.length:
            i, t = n - 2, 1.0
        else:
            i = bisect.bisect_right(cum, s) - 1
            i = max(0, min(n - 2, i))
            seg = cum[i + 1] - cum[i]
            t = (s - cum[i]) / seg if seg > 1e-9 else 0.0
        a, b = pts[i], pts[i + 1]
        h0, h1 = self.vh[i], self.vh[i + 1]
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                h0 + normalize_angle(h1 - h0) * t)


class Crosswalk:
    def __init__(self, cx, cy, ux, uy, vx, vy, road_half, signal, road_axis, kind):
        self.cx, self.cy = cx, cy
        self.ux, self.uy = ux, uy
        self.vx, self.vy = vx, vy
        self.road_half = road_half
        self.wait_off = road_half + 1.3
        self.signal = signal
        self.road_axis = road_axis
        self.walk_axis = 'NS' if road_axis == 'EW' else 'EW'
        self.kind = kind
        self.ped = None
        self.timer = random.uniform(1.0, 14.0)


class Node:
    def __init__(self, i, j, x, y):
        self.i, self.j = i, j
        self.x, self.y = x, y
        self.legs = {}
        self.kind = None
        self.ext = {}
        self.ex = 0.0
        self.ey = 0.0
        self.signal = None
        self.in_lanes = {}
        self.out_lanes = {}
        self.conns = []
        self.occ = []
        self.row_rt = None
        self.col_rt = None


class Edge:
    def __init__(self, a, b, d, rt, line):
        self.a, self.b, self.d = a, b, d
        self.axis = axis_of(d)
        self.rt = rt
        self.half = rt.half
        self.kind = rt.key
        self.line = line                      # ('row', j) / ('col', i)
        self.protected = False
        self.p0 = self.p1 = None
        self.length = 0.0
        self.lanes = {}
        self.cl = None
        self.cum = None
        self.elev = None
        self.curve_amp = 0.0
        self.oneway_dir = None
        self.midblock = None

    def pose(self, s, lat=0.0):
        x, y, h = pl_point(self.cl, self.cum, s)
        if lat:
            x += -math.sin(h) * lat
            y += math.cos(h) * lat
        return x, y, h


class Building:
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'h', 'color', 'variant', 'bid', 'cx', 'cy', 'rad', 'wall', 'poly', 'landmark')

    def __init__(self, x0, y0, x1, y1, h, color, variant, bid, wall=False):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.h, self.color, self.variant, self.bid, self.wall = h, color, variant, bid, wall
        self.cx, self.cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        self.rad = math.hypot(x1 - x0, y1 - y0) * 0.5
        self.poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.landmark = None


COL_SIDEWALK = (152, 150, 144)
COL_SHOULDER = (120, 118, 112)
COL_WHITE = (236, 236, 230)
COL_YELLOW = (236, 196, 40)
COL_GRASS = (62, 124, 52)
COL_MEDIAN = (84, 128, 72)
COL_CURB = (196, 194, 188)
COL_CONCRETE = (172, 170, 164)

RAMP_LEN = 66.0
DECK_H = 6.8
DECK_T = 1.3


# =============================================================================
# THE CITY  (random each seed)
# =============================================================================
class CityMap:
    """
    Random city generator: varied road types (highway, avenue, street, narrow lane,
    one-way), winding roads, T-junctions, corners, flyovers with ramps, signals,
    crosswalks, buildings, trees, lamps.
    """

    def __init__(self, seed=None):
        self.seed = int(seed if seed is not None else random.randrange(1, 10 ** 6))
        self.rng = random.Random(self.seed)
        random.seed(self.seed)
        self.no_oneway = set()
        attempt = 0
        while True:
            attempt += 1
            try:
                self._generate(attempt)
                break
            except _Regen as r:
                if attempt > 12:
                    raise RuntimeError("city generation failed: " + str(r))
                continue
        self.stats = (len(self.nodes), len(self.edges), len(self.lanes), len(self.crosswalks),
                      len(self.buildings), len(self.flyovers))

    # ------------------------------------------------------------------
    def _generate(self, attempt):
        Path._ids = 0
        C = Config
        self.nodes = {}
        self.edges = []
        self.lanes = []
        self.crosswalks = []
        self.signal_heads = []
        self.col_hash = SpatialHash(60.0)
        self.buildings = []
        self.circles = []            # (x, y, r, kind, height, lvl)
        self.barriers = []           # (ax, ay, bx, by, half_thick, height, kind, poly, lvl, visible)
        self.ground_hash = SpatialHash(80.0)
        self.bld_hash = SpatialHash(100.0)
        self.prop_hash = SpatialHash(80.0)
        self.surf_hash = SpatialHash(60.0)
        self.deck_hash = SpatialHash(80.0)
        self.trees = []
        self.lamps = []
        self.pillars = []
        self.decks = []
        self.flyovers = []
        self.bends = []
        self.spawns = []
        self.signs = []             # Road signs: Speed limit, Zebra crossing, Stop
        self._layout(attempt)
        self._build_graph(attempt)
        self._shape_edges(attempt)
        self._build_lanes()
        self._build_connectors()
        self._check_oneway()
        self._build_signals_and_crosswalks()
        self._build_ground()
        self._build_bridges()
        self._build_barriers()
        self._build_furniture()
        self._build_buildings()
        self._clean_signs()
        self._index_static()
        self._build_nav()
        self._build_spawns()

    def _clean_signs(self):
        """Sanity filter: guarantees no sign is placed on road asphalt, middle of road, or inside crosswalks."""
        valid = []
        for s in self.signs:
            sx, sy = s['x'], s['y']
            fx, fy = s.get('fx', 0.0), s.get('fy', 0.0)

            # Check 8-point perimeter with 1.4m clearance to ensure pole is not on or touching asphalt
            def near_road(x, y, rad=1.4):
                for dx, dy in [(0, 0), (rad, 0), (-rad, 0), (0, rad), (0, -rad),
                               (rad * 0.7, rad * 0.7), (-rad * 0.7, rad * 0.7),
                               (rad * 0.7, -rad * 0.7), (-rad * 0.7, -rad * 0.7)]:
                    surf, _ = self.surface_at(x + dx, y + dy)
                    if surf in (1, 2):
                        return True
                return False

            # Check crosswalk footprint clearance
            def near_crosswalk(x, y):
                for cw in getattr(self, 'crosswalks', []):
                    dx, dy = x - cw.cx, y - cw.cy
                    along = abs(dx * cw.ux + dy * cw.uy)
                    across = abs(dx * cw.vx + dy * cw.vy)
                    if along < 3.2 and across < (cw.road_half + 1.2):
                        return True
                return False

            # If touching road, try displacing outward along sign normal away from road center
            if near_road(sx, sy) or near_crosswalk(sx, sy):
                displaced = False
                for dist in (1.5, 2.2, 3.0):
                    for sign_dir in (1.0, -1.0):
                        test_x = sx + (-fy) * dist * sign_dir
                        test_y = sy + (fx) * dist * sign_dir
                        if not near_road(test_x, test_y, 1.2) and not near_crosswalk(test_x, test_y):
                            s['x'], s['y'] = test_x, test_y
                            sx, sy = test_x, test_y
                            displaced = True
                            break
                    if displaced:
                        break
                if not displaced:
                    continue  # Discard if cannot be placed safely on sidewalk

            valid.append(s)
        self.signs = valid

    # ------------------------------------------------------------ layout
    def _layout(self, attempt):
        C = Config
        rng = random.Random(self.seed * 7919 + 13)
        N = C.GRID_N
        self.N = N
        xs, ys = [0.0], [0.0]
        for _ in range(N - 1):
            xs.append(xs[-1] + rng.uniform(C.BLOCK_MIN, C.BLOCK_MAX))
            ys.append(ys[-1] + rng.uniform(C.BLOCK_MIN, C.BLOCK_MAX))
        self.xs, self.ys = xs, ys
        cx, cy = rng.choice([3, 4, 5]), rng.choice([3, 4, 5])
        ac, bc = rng.randint(2, cx - 1), rng.randint(cx + 1, 6)
        ar, br = rng.randint(2, cy - 1), rng.randint(cy + 1, 6)
        self.cx, self.cy, self.ac, self.bc, self.ar, self.br = cx, cy, ac, bc, ar, br
        col_t, row_t = [None] * N, [None] * N
        for k in (0, N - 1):
            col_t[k] = row_t[k] = 'H'
        col_t[cx] = 'H' if rng.random() < 0.8 else 'A'
        row_t[cy] = 'H' if rng.random() < 0.8 else 'A'
        for c in (ac, bc):
            col_t[c] = 'A' if rng.random() < 0.4 else 'S'
        for r_ in (ar, br):
            row_t[r_] = 'A' if rng.random() < 0.4 else 'S'
        col_dir, row_dir = {}, {}
        for k in range(N):
            for arr, dirs, name in ((col_t, col_dir, 'col'), (row_t, row_dir, 'row')):
                if arr[k] is None:
                    r = rng.random()
                    t = 'S' if r < 0.5 else ('L' if r < 0.8 else 'O')
                    if t == 'O' and (name, k) in self.no_oneway:
                        t = 'S'
                    arr[k] = t
                    if t == 'O':
                        dirs[k] = rng.choice('NS' if name == 'col' else 'EW')
        self.col_t, self.row_t, self.col_dir, self.row_dir = col_t, row_t, col_dir, row_dir

        self.row_nodes, self.col_nodes = {}, {}
        for j in range(N):
            if j in (0, N - 1) or j == cy:
                self.row_nodes[j] = [[0, ac, cx, bc, N - 1]]
            elif j in (ar, br):
                self.row_nodes[j] = [list(range(N))]
            else:
                self.row_nodes[j] = [list(range(1, cx)), list(range(cx + 1, N - 1))]
        for i in range(N):
            if i in (0, N - 1) or i == cx:
                self.col_nodes[i] = [[0, ar, cy, br, N - 1]]
            elif i in (ac, bc):
                self.col_nodes[i] = [list(range(N))]
            else:
                self.col_nodes[i] = [list(range(1, cy)), list(range(cy + 1, N - 1))]

        # --- flyovers: grade-separated crossings of two major roads
        cands = [(cx, cy)]
        extra = [(ac, cy), (bc, cy), (cx, ar), (cx, br)]
        rng.shuffle(extra)
        cands.append(extra[0])
        if rng.random() < 0.5:
            cands.append(extra[1])
        self.fly_plan = []
        over_set, under_set = set(), set()
        for (i, j) in cands:
            ranks = RT[col_t[i]].rank, RT[row_t[j]].rank
            if ranks[0] < 2 and ranks[1] < 2:
                continue
            over = 'col' if ranks[0] <= ranks[1] else 'row'
            if (i, j) == (cx, cy):
                over = 'col'
            # bridging line is the one passing over; it must have long neighbours
            if over == 'col':
                lst = self.col_nodes[i][0]
                k = lst.index(j) if j in lst else -1
                if k <= 0 or k >= len(lst) - 1:
                    continue
                span = abs(ys[lst[k + 1]] - ys[lst[k - 1]])
                under_half = RT[row_t[j]].half + RT[row_t[j]].side_w
            else:
                lst = self.row_nodes[j][0]
                k = lst.index(i) if i in lst else -1
                if k <= 0 or k >= len(lst) - 1:
                    continue
                span = abs(xs[lst[k + 1]] - xs[lst[k - 1]])
                under_half = RT[col_t[i]].half + RT[col_t[i]].side_w
            if span < 2 * RAMP_LEN + 2 * under_half + 40:
                continue
            if over == 'col':
                d_before = abs(ys[j] - ys[lst[k - 1]])
                d_after = abs(ys[lst[k + 1]] - ys[j])
            else:
                d_before = abs(xs[i] - xs[lst[k - 1]])
                d_after = abs(xs[lst[k + 1]] - xs[i])
            need = RAMP_LEN + under_half + 14.0 + 11.0
            if d_before < need or d_after < need:
                continue
            o_line = ('col', i) if over == 'col' else ('row', j)
            u_line = ('row', j) if over == 'col' else ('col', i)
            if o_line in over_set or o_line in under_set or u_line in over_set:
                continue
            over_set.add(o_line)
            under_set.add(u_line)
            self.fly_plan.append((i, j, over))
            self.col_nodes[i][0] = [v for v in self.col_nodes[i][0] if v != j]
            self.row_nodes[j][0] = [v for v in self.row_nodes[j][0] if v != i]

    # ------------------------------------------------------------- graph
    def _node(self, i, j):
        if (i, j) not in self.nodes:
            n = Node(i, j, self.xs[i], self.ys[j])
            n.row_rt, n.col_rt = RT[self.row_t[j]], RT[self.col_t[i]]
            self.nodes[(i, j)] = n
        return self.nodes[(i, j)]

    def _build_graph(self, attempt):
        C = Config
        N = self.N
        rng = random.Random(self.seed * 104729 + attempt)
        bridging = {}
        for (i, j, over) in self.fly_plan:
            bridging[(over, i if over == 'col' else j)] = True
        edges = []
        fly_lines = {}
        for (i, j, over) in self.fly_plan:
            fly_lines.setdefault(('col', i) if over == 'col' else ('row', j), []).append((i, j))
        for j in range(N):
            for seg in self.row_nodes[j]:
                lst = sorted(seg)
                for a, b in zip(lst, lst[1:]):
                    edges.append(((a, j), (b, j), 'E', RT[self.row_t[j]], ('row', j)))
        for i in range(N):
            for seg in self.col_nodes[i]:
                lst = sorted(seg)
                for a, b in zip(lst, lst[1:]):
                    edges.append(((i, a), (i, b), 'S', RT[self.col_t[i]], ('col', i)))
        protected = {('col', self.ac), ('col', self.bc), ('row', self.ar), ('row', self.br)}
        # --- random removal of minor street segments (variety), keeping the network sane
        removable = [k for k, ed in enumerate(edges)
                     if ed[3].rank <= 1 and ed[4] not in protected and ed[4] not in fly_lines]
        rng.shuffle(removable)
        removed = set()
        legs_of = defaultdict(set)
        for ed in edges:
            legs_of[ed[0]].add(ed[2])
            legs_of[ed[1]].add(OPP[ed[2]])

        def ok_legs(s):
            if len(s) >= 3:
                return True
            if len(s) == 2:
                a, b = tuple(s)
                return OPP[a] != b
            return False
        for k in removable[:int(len(removable) * 0.16)]:
            ka, kb, d, rt, line = edges[k]
            sa, sb = set(legs_of[ka]), set(legs_of[kb])
            sa.discard(d)
            sb.discard(OPP[d])
            if ok_legs(sa) and ok_legs(sb):
                legs_of[ka], legs_of[kb] = sa, sb
                removed.add(k)
        kept = [ed for k, ed in enumerate(edges) if k not in removed]
        if not self._connected(kept):
            kept = edges
        for (ka, kb, d, rt, line) in kept:
            a, b = self._node(*ka), self._node(*kb)
            e = Edge(a, b, d, rt, line)
            if rt.oneway:
                e.oneway_dir = self.row_dir[line[1]] if line[0] == 'row' else self.col_dir[line[1]]
            e.protected = line in protected
            a.legs[d] = e
            b.legs[OPP[d]] = e
            self.edges.append(e)
        for n in self.nodes.values():
            assert len(n.legs) >= 2, f"dead end at {n.i},{n.j}"
            n.kind = 'bend' if len(n.legs) == 2 else 'junction'
            if n.kind == 'bend':
                rk = max(RT[self.row_t[n.j]].rank, RT[self.col_t[n.i]].rank)
                d = C.BEND_D_HIGHWAY if rk >= 3 else (55.0 if rk == 2 else (C.BEND_D_STREET if rk == 1 else 14.0))
                for leg in n.legs:
                    n.ext[leg] = d
            else:
                n.ex = n.col_rt.half
                n.ey = n.row_rt.half
                for dd in 'ESWN':
                    n.ext[dd] = n.ex if dd in 'EW' else n.ey
        for e in self.edges:
            ux, uy = U[e.d]
            e.p0 = (e.a.x + ux * e.a.ext[e.d], e.a.y + uy * e.a.ext[e.d])
            ob = OPP[e.d]
            e.p1 = (e.b.x - ux * e.b.ext[ob], e.b.y - uy * e.b.ext[ob])
            e.length = math.hypot(e.p1[0] - e.p0[0], e.p1[1] - e.p0[1])
            if e.length < 20.0:
                raise _Regen("short edge")
        # bridging edges
        self.fly_edges = []
        for (i, j, over) in self.fly_plan:
            line = ('col', i) if over == 'col' else ('row', j)
            x, y = self.xs[i], self.ys[j]
            best = None
            for e in self.edges:
                if e.line != line:
                    continue
                lo = min(e.p0[0], e.p1[0]) - 0.5, min(e.p0[1], e.p1[1]) - 0.5
                hi = max(e.p0[0], e.p1[0]) + 0.5, max(e.p0[1], e.p1[1]) + 0.5
                if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
                    best = e
            if best is None:
                continue
            u = math.hypot(x - best.p0[0], y - best.p0[1])
            under_rt = RT[self.row_t[j]] if over == 'col' else RT[self.col_t[i]]
            uh = under_rt.half + under_rt.side_w
            if best.length < 2 * RAMP_LEN + 2 * uh + 40 or u < RAMP_LEN + uh + 14 or u > best.length - RAMP_LEN - uh - 14:
                raise _Regen("flyover geometry")
            best.elev = Elev(DECK_H, RAMP_LEN, best.length)
            self.fly_edges.append(best)
            self.flyovers.append(dict(i=i, j=j, x=x, y=y, over=best, u=u, under_rt=under_rt))

    @staticmethod
    def _connected(edges):
        adj = defaultdict(set)
        for ka, kb, *_ in edges:
            adj[ka].add(kb)
            adj[kb].add(ka)
        if not adj:
            return False
        start = next(iter(adj))
        seen = {start}
        st = [start]
        while st:
            v = st.pop()
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    st.append(w)
        return len(seen) == len(adj)

    # ------------------------------------------------------ curved roads
    def _shape_edges(self, attempt):
        rng = random.Random(self.seed * 15485863 + attempt)
        LS = 24.0
        rreq = {3: 150.0, 2: 85.0, 1: 34.0, 0: 26.0}
        for e in self.edges:
            L = e.length
            ux, uy = U[e.d]
            rx, ry = right_vec(e.d)
            n = max(2, int(L / 3.0) + 1)
            amp = 0.0
            shape = 0
            if e.elev is None and L - 2 * LS >= 44.0 and rng.random() < (0.5 if e.rt.rank >= 1 else 0.6):
                M = L - 2 * LS
                amax = min(40.0 if e.rt.rank >= 2 else 26.0, M * 0.35)
                amp = rng.uniform(0.35, 1.0) * amax * rng.choice((-1, 1))
                shape = rng.choice((0, 0, 1))

            def build(a_):
                pts = []
                for k in range(n + 1):
                    s = L * k / n
                    u_ = clamp((s - LS) / max(L - 2 * LS, 1.0), 0.0, 1.0)
                    if shape == 0:
                        w = a_ * math.sin(math.pi * u_) ** 2
                    else:
                        w = a_ * 1.7 * math.sin(2 * math.pi * u_) * math.sin(math.pi * u_) ** 2
                    pts.append((e.p0[0] + ux * s + rx * w, e.p0[1] + uy * s + ry * w))
                return pts
            if amp != 0.0:
                pts = build(amp)
                tries = 0
                while pl_min_radius(pts) < rreq[e.rt.rank] and tries < 12:
                    amp *= 0.82
                    pts = build(amp)
                    tries += 1
                if pl_min_radius(pts) < rreq[e.rt.rank] or abs(amp) < 3.0:
                    amp = 0.0
            if amp == 0.0:
                pts = [e.p0, e.p1] if e.elev is not None else [(e.p0[0] + ux * L * k / n, e.p0[1] + uy * L * k / n) for k in range(n + 1)]
                if e.elev is None:
                    pts[-1] = e.p1
            e.cl = pts
            e.cum = pl_cum(pts)
            e.length = e.cum[-1]
            e.curve_amp = amp
            if e.elev is not None:
                e.elev = Elev(DECK_H, RAMP_LEN, e.length)

    # ------------------------------------------------------------------ lanes
    @staticmethod
    def _lead(node):
        return Config.LEAD if node.kind == 'junction' else 0.0

    def _build_lanes(self):
        for e in self.edges:
            rt = e.rt
            L = e.length
            dirs = [e.d, OPP[e.d]] if not rt.oneway else [e.oneway_dir]
            for t in dirs:
                fwd = (t == e.d)
                sn, en = (e.a, e.b) if fwd else (e.b, e.a)
                ls, le = self._lead(sn), self._lead(en)
                if fwd:
                    base = pl_slice(e.cl, e.cum, ls, L - le)
                    u0, du = ls, 1.0
                else:
                    base = pl_slice(e.cl, e.cum, le, L - ls)[::-1]
                    u0, du = L - ls, -1.0
                lanes = []
                for k, o in enumerate(rt.offs):
                    lane = Path(pl_offset(base, o), 'lane', rt.limit)
                    lane.edge, lane.tdir, lane.idx, lane.node = e, t, k, en
                    lane.axis = axis_of(t)
                    lane.elev, lane.u0, lane.du = e.elev, u0, du
                    lanes.append(lane)
                    self.lanes.append(lane)
                e.lanes[t] = lanes
                sn.out_lanes[t] = lanes
                en.in_lanes[t] = lanes

    # ------------------------------------------------------------- connectors
    def bend_curve(self, node, t_in, t_out, o, n=20):
        d = node.ext[OPP[t_in]]
        u1 = U[t_in]
        u2 = U[t_out]
        r1 = right_vec(t_in)
        r2 = right_vec(t_out)
        E = (node.x - u1[0] * d + r1[0] * o, node.y - u1[1] * d + r1[1] * o)
        X = (node.x + u2[0] * d + r2[0] * o, node.y + u2[1] * d + r2[1] * o)
        k = (X[0] - E[0]) * u1[0] + (X[1] - E[1]) * u1[1]
        Cc = (E[0] + u1[0] * k, E[1] + u1[1] * k)
        return bezier2(E, Cc, X, n)

    def _build_connectors(self):
        for n in self.nodes.values():
            n.conns = []
            for t_in, lanes_in in n.in_lanes.items():
                for lane in lanes_in:
                    if n.kind == 'bend':
                        leg = [d for d in n.legs if d != OPP[t_in]][0]
                        if leg not in n.out_lanes:
                            continue
                        options = [('R' if RIGHT[t_in] == leg else 'L', leg)]
                    else:
                        options = [(m, td) for m, td in (('S', t_in), ('R', RIGHT[t_in]), ('L', LEFT[t_in]))
                                   if td in n.out_lanes]
                        # U-turn connector: allowed at junctions from innermost lane to opposite leg
                        if lane == lanes_in[-1] and OPP[t_in] in n.out_lanes:
                            options.append(('U', OPP[t_in]))
                    combos = []
                    for move, t_out in options:
                        outs = n.out_lanes[t_out]
                        if n.kind == 'bend' or move == 'S':
                            combos.append((move, t_out, outs[min(lane.idx, len(outs) - 1)]))
                        elif move == 'U':
                            combos.append((move, t_out, outs[0]))
                            if len(outs) > 1:
                                combos.append((move, t_out, outs[-1]))
                        else:
                            for o_ in ([outs[0], outs[-1]] if len(outs) > 1 else outs):
                                combos.append((move, t_out, o_))
                    for move, t_out, out in combos:
                        E, X = lane.pts[-1], out.pts[0]
                        if move == 'S':
                            pts = [E, X]
                            if abs(E[0] - X[0]) + abs(E[1] - X[1]) < 0.05:
                                pts = [E, (X[0] + 0.05, X[1] + 0.05)]
                        elif move == 'U':
                            u1 = U[t_in]
                            d_reach = max(6.0, min(n.ext[t_in], 14.0) * 0.72 + 2.0)
                            C1 = (E[0] + u1[0] * d_reach, E[1] + u1[1] * d_reach)
                            C2 = (X[0] + u1[0] * d_reach, X[1] + u1[1] * d_reach)
                            pts = bezier3(E, C1, C2, X, 24)
                        else:
                            u1 = U[t_in]
                            k = (X[0] - E[0]) * u1[0] + (X[1] - E[1]) * u1[1]
                            Cc = (E[0] + u1[0] * k, E[1] + u1[1] * k)
                            pts = bezier2(E, Cc, X, 24 if lane.edge.rt.rank >= 2 else 12)
                        conn = Path(pts, 'conn', lane.limit)
                        conn.move, conn.in_lane, conn.out_lane, conn.node = move, lane, out, n
                        conn.tdir = t_in
                        conn.edge = lane.edge
                        if move != 'S':
                            conn.rmin = pl_min_radius(pts)
                            conn.cap = clamp(math.sqrt(3.2 * conn.rmin), 3.0, 5.5 if move == 'U' else lane.limit)
                        conn.out = [out]
                        conn.dups = sum(1 for cm_ in combos if cm_[0] == move)
                        lane.out.append(conn)
                        n.conns.append(conn)
            if n.kind == 'bend':
                self.bends.append(n)
        self._build_conflicts()

    def _build_conflicts(self):
        def dense(pts, step=1.0):
            out = [pts[0]]
            for i in range(1, len(pts)):
                a, b = pts[i - 1], pts[i]
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                k = max(1, int(L / step))
                for j in range(1, k + 1):
                    out.append((a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k))
            return out
        for n in self.nodes.values():
            n.occ = []
            for c in n.conns:
                c.conflicts = set()
                c.conflicts_wide = set()
            if n.kind != 'junction':
                continue
            dp = {c.pid: dense(c.pts) for c in n.conns}
            for i, a in enumerate(n.conns):
                for b in n.conns[i + 1:]:
                    if a.in_lane is b.in_lane:
                        continue
                    pa, pb = dp[a.pid], dp[b.pid]
                    close = False
                    for (x1, y1) in pa:
                        for (x2, y2) in pb:
                            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 6.76:
                                close = True
                                break
                        if close:
                            break
                    wide = close
                    if not close:
                        for (x1, y1) in pa[::2]:
                            for (x2, y2) in pb[::2]:
                                if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 19.0:
                                    wide = True
                                    break
                            if wide:
                                break
                    if close:
                        a.conflicts.add(b.pid)
                        b.conflicts.add(a.pid)
                    if wide:
                        a.conflicts_wide.add(b.pid)
                        b.conflicts_wide.add(a.pid)

    def _check_oneway(self):
        bad = [ln for ln in self.lanes if not ln.out]
        if not bad:
            return
        lines = set()
        for ln in bad:
            for e in list(ln.node.legs.values()) + [ln.edge]:
                if e.rt.oneway:
                    lines.add(e.line)
        if not lines:
            raise RuntimeError("lane without exit that is not caused by a one-way street")
        self.no_oneway |= lines
        raise _Regen("one-way conflict")

    # --------------------------------------------------- signals & crosswalks
    def _build_signals_and_crosswalks(self):
        C = Config
        rng = random.Random(self.seed * 31 + 5)
        green = {3: 18.0, 2: 15.0, 1: 11.0, 0: 9.0}
        for n in self.nodes.values():
            if n.kind != 'junction':
                continue
            n.signal = Signal(green[n.row_rt.rank], green[n.col_rt.rank], 0.0)
            n.signal.offset = rng.uniform(0.0, n.signal.cycle)
            for t_in, lanes in n.in_lanes.items():
                for ln in lanes:
                    ln.signal = n.signal
                    ln.stop_s = ln.length - 0.5
                fx, fy = U[t_in]
                rx, ry = right_vec(t_in)
                p_stop = (lanes[0].pts[-1][0] - fx * 0.5, lanes[0].pts[-1][1] - fy * 0.5)
                # Primary signal: at the near-side stop line on the curb beside approaching driver
                curb_off = lanes[0].edge.rt.half + 1.2
                nx = p_stop[0] + rx * curb_off - fx * 1.0
                ny = p_stop[1] + ry * curb_off - fy * 1.0
                self.signal_heads.append(dict(x=nx, y=ny, fx=-fx, fy=-fy, axis=axis_of(t_in), sig=n.signal,
                                              stop=p_stop, tdir=t_in, primary=True))
                self.circles.append((nx, ny, 0.25, 'pole', 5.6, 0))
                # Secondary signal: far-side across the junction for long-range visibility
                lat = n.ext[RIGHT[t_in]] + 1.3
                extra = 1.2 if max(n.row_rt.rank, n.col_rt.rank) >= 2 else 0.5
                hx = n.x + fx * (n.ext[t_in] + 1.2 + extra) + rx * (lat + 1.4 + extra + 0.4)
                hy = n.y + fy * (n.ext[t_in] + 1.2 + extra) + ry * (lat + 1.4 + extra + 0.4)
                self.signal_heads.append(dict(x=hx, y=hy, fx=-fx, fy=-fy, axis=axis_of(t_in), sig=n.signal,
                                              stop=p_stop, tdir=t_in, primary=False))
                self.circles.append((hx, hy, 0.25, 'pole', 5.6, 0))
            for d, e in n.legs.items():
                if e.rt.rank > 2:
                    continue
                ux, uy = U[d]
                vx, vy = right_vec(d)
                dist = n.ext[d] + C.LEAD - 3.0
                # Zebra Crossing Warning Sign (placed on right sidewalk 14m before crosswalk)
                try:
                    zw_dist = n.ext[d] + C.LEAD + 14.0
                    zw_side = (e.rt.half + 1.8)
                    zw_x = n.x + ux * zw_dist + vx * zw_side
                    zw_y = n.y + uy * zw_dist + vy * zw_side
                    self.signs.append({
                        'x': zw_x, 'y': zw_y, 'z': 0.0, 'h': 2.7,
                        'fx': ux, 'fy': uy,
                        'kind': 'zebra', 'val': 25
                    })
                except Exception:
                    pass
                self.crosswalks.append(Crosswalk(n.x + ux * dist, n.y + uy * dist, ux, uy, vx, vy,
                                                 e.rt.half, n.signal, axis_of(d), 'junction'))
        for e in self.edges:
            if e.rt.rank <= 1 and e.elev is None and e.length >= 78.0:
                x, y, h = e.pose(e.length * 0.5)
                cw = Crosswalk(x, y, math.cos(h), math.sin(h), -math.sin(h), math.cos(h),
                               e.rt.half, None, e.axis, 'midblock')
                self.crosswalks.append(cw)
                e.midblock = cw

    # ----------------------------------------------------------- ground world
    def _gadd(self, layer, color, poly, maxd=1e9):
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        rad = max(math.hypot(p[0] - cx, p[1] - cy) for p in poly)
        it = (layer, color, poly, cx, cy, rad, maxd)
        self.ground_hash.insert(it, cx - rad, cy - rad, cx + rad, cy + rad)

    @staticmethod
    def _rect_poly(p, u, r, s0, s1, o0, o1):
        return [(p[0] + u[0] * s0 + r[0] * o0, p[1] + u[1] * s0 + r[1] * o0),
                (p[0] + u[0] * s1 + r[0] * o0, p[1] + u[1] * s1 + r[1] * o0),
                (p[0] + u[0] * s1 + r[0] * o1, p[1] + u[1] * s1 + r[1] * o1),
                (p[0] + u[0] * s0 + r[0] * o1, p[1] + u[1] * s0 + r[1] * o1)]

    def _surf(self, poly, code, limit, lvl):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        self.surf_hash.insert((poly, code, limit, lvl), min(xs), min(ys), max(xs), max(ys))

    def _line(self, sub, o, hwid, color, maxd=1e9):
        self._gadd(2, color, pl_strip(sub, o - hwid, o + hwid), maxd)

    def _build_ground(self):
        C = Config
        for e in self.edges:
            if e.elev is not None:
                continue
            rt = e.rt
            L = e.length
            hw, sw = rt.half, rt.side_w
            code = 2 if rt.rank >= 2 else 1
            n_ch = max(1, int(math.ceil(L / 28.0)))
            for k in range(n_ch):
                s0 = L * k / n_ch
                s1 = min(L, L * (k + 1) / n_ch + 0.5)
                sub = pl_slice(e.cl, e.cum, s0, s1)
                for (o0, o1) in ((-(hw + sw), -hw), (hw, hw + sw)):
                    poly = pl_strip(sub, o0, o1)
                    self._gadd(0, rt.side_col, poly)
                    self._surf(poly, 3, 0, 0)
                road = pl_strip(sub, -hw, hw)
                self._gadd(1, rt.road_col, road)
                self._surf(road, code, rt.limit_kmh, 0)
                st = rt.style
                if st == 'divided':
                    self._gadd(2, (112, 110, 106), pl_strip(sub, -rt.median, rt.median))
                    for o in (-(rt.median + 0.12), rt.median + 0.12):
                        self._line(sub, o, 0.09, COL_YELLOW)
                    for o in (-(hw - 0.3), hw - 0.3):
                        self._line(sub, o, 0.09, COL_WHITE)
                elif st == 'avenue':
                    self._gadd(2, COL_MEDIAN, pl_strip(sub, -rt.median, rt.median))
                    for o in (-rt.median, rt.median):
                        self._line(sub, o, 0.07, COL_CURB)
                    for o in (-(hw - 0.25), hw - 0.25):
                        self._line(sub, o, 0.09, COL_WHITE)
                elif st == 'street':
                    for o in (-0.17, 0.17):
                        self._line(sub, o, 0.07, COL_YELLOW)
                elif st == 'oneway':
                    for o in (-(hw - 0.25), hw - 0.25):
                        self._line(sub, o, 0.09, COL_WHITE)
            # dashed lane dividers + arrows
            dash_offs = {'divided': (-4.6, 4.6), 'avenue': (-4.05, 4.05), 'oneway': (0.0,)}.get(rt.style, ())
            for o in dash_offs:
                s = 3.0
                while s + 3.0 < L - 1.0:
                    sub = pl_slice(e.cl, e.cum, s, s + 3.0)
                    self._line(sub, o, 0.08, COL_WHITE, 110.0)
                    s += 9.0
            if rt.oneway:
                t = e.oneway_dir
                s = 40.0
                while s < L - 30.0:
                    x, y, h = e.pose(s)
                    if t != e.d:
                        h += math.pi
                    fx, fy = math.cos(h), math.sin(h)
                    rx, ry = -fy, fx
                    for lo in rt.offs:
                        lat = lo if t == e.d else -lo
                        ex_, ey_, _h = e.pose(s, lat)
                        pts = [(-1.6, -0.18), (0.4, -0.18), (0.4, -0.7), (2.0, 0.0), (0.4, 0.7), (0.4, 0.18), (-1.6, 0.18)]
                        poly = [(ex_ + fx * a_ + rx * b_, ey_ + fy * a_ + ry * b_) for a_, b_ in pts]
                        self._gadd(2, COL_WHITE, poly, 120.0)
                    s += 55.0

        # ---------------- junction boxes
        for n in self.nodes.values():
            rk, ck = n.row_rt, n.col_rt
            if n.kind == 'junction':
                sx, sy = ck.side_w, rk.side_w
                plate = [(n.x - n.ex - sx, n.y - n.ey - sy), (n.x + n.ex + sx, n.y - n.ey - sy),
                         (n.x + n.ex + sx, n.y + n.ey + sy), (n.x - n.ex - sx, n.y + n.ey + sy)]
                lowr = rk if rk.rank <= ck.rank else ck
                hir = rk if rk.rank >= ck.rank else ck
                self._gadd(0, lowr.side_col, plate)
                self._surf(plate, 3, 0, 0)
                box = [(n.x - n.ex, n.y - n.ey), (n.x + n.ex, n.y - n.ey),
                       (n.x + n.ex, n.y + n.ey), (n.x - n.ex, n.y + n.ey)]
                self._gadd(1, hir.road_col, box)
                code = 2 if hir.rank >= 2 else 1
                self._surf(box, code, hir.limit_kmh, 0)
                for cn in n.conns:
                    if cn.move == 'S':
                        continue
                    for i in range(0, len(cn.pts) - 1):
                        seg = strip_polygon([cn.pts[i], cn.pts[i + 1]], 2.5, 2.5)
                        self._gadd(1, hir.road_col, seg)
                        self._surf(seg, code, hir.limit_kmh, 0)
                for t_in, lanes in n.in_lanes.items():
                    e = lanes[0].edge
                    ux, uy = U[t_in]
                    rx, ry = right_vec(t_in)
                    pe = lanes[0].pts[-1]
                    sp = (pe[0] - ux * 0.5, pe[1] - uy * 0.5)
                    offs = e.rt.offs
                    lo, hi = min(offs) - 1.25, max(offs) + 1.25
                    if not e.rt.oneway:
                        lo = max(lo, e.rt.median + 0.2)
                    self._gadd(3, COL_WHITE, self._rect_poly(sp, (ux, uy), (rx, ry), -0.25, 0.25, lo, hi), 170.0)
            else:
                legs = list(n.legs)
                t_in, t_out = OPP[legs[0]], legs[1]
                ea, eb = n.legs[legs[0]], n.legs[legs[1]]
                rta = ea.rt if ea.rt.rank >= eb.rt.rank else eb.rt
                hw = max(ea.rt.half, eb.rt.half)
                sw = max(ea.rt.side_w, eb.rt.side_w)
                nn = 26 if rta.rank >= 2 else 14
                lft = self.bend_curve(n, t_in, t_out, -hw, nn)
                rgt = self.bend_curve(n, t_in, t_out, hw, nn)
                self._gadd(1, rta.road_col, lft + rgt[::-1])
                pl_ = self.bend_curve(n, t_in, t_out, -(hw + sw), nn)
                pr_ = self.bend_curve(n, t_in, t_out, hw + sw, nn)
                plate = pl_ + pr_[::-1]
                self._gadd(0, rta.side_col if rta.rank < 3 else COL_SHOULDER, plate)
                code = 2 if rta.rank >= 2 else 1
                self._surf(lft + rgt[::-1], code, rta.limit_kmh, 0)
                self._surf(plate, 3, 0, 0)
                if rta.rank >= 2:
                    mh = rta.median
                    for o in (-(mh + 0.12), mh + 0.12):
                        c = self.bend_curve(n, t_in, t_out, o, nn)
                        for i in range(len(c) - 1):
                            self._gadd(2, COL_YELLOW, strip_polygon([c[i], c[i + 1]], 0.09, 0.09))
                    do = 4.6 if rta.rank >= 3 else 4.05
                    for o in (-do, do):
                        c = self.bend_curve(n, t_in, t_out, o, nn)
                        for i in range(0, len(c) - 1, 2):
                            self._gadd(2, COL_WHITE, strip_polygon([c[i], c[i + 1]], 0.08, 0.08), 110.0)
                    for o in (-(hw - 0.3), hw - 0.3):
                        c = self.bend_curve(n, t_in, t_out, o, nn)
                        for i in range(len(c) - 1):
                            self._gadd(2, COL_WHITE, strip_polygon([c[i], c[i + 1]], 0.09, 0.09))
                elif rta.rank == 1:
                    for o in (-0.17, 0.17):
                        c = self.bend_curve(n, t_in, t_out, o, nn)
                        for i in range(len(c) - 1):
                            self._gadd(2, COL_YELLOW, strip_polygon([c[i], c[i + 1]], 0.07, 0.07))

        # ---------------- crosswalk zebras
        for cw in self.crosswalks:
            r = (cw.vx, cw.vy)
            u = (cw.ux, cw.uy)
            o = -cw.road_half + 0.35
            while o + 0.55 < cw.road_half - 0.2:
                self._gadd(3, COL_WHITE, self._rect_poly((cw.cx, cw.cy), u, r, -1.5, 1.5, o, o + 0.55), 130.0)
                o += 1.1
            if cw.kind == 'midblock':
                for sgn in (1, -1):
                    sp = (cw.cx - u[0] * sgn * 3.6, cw.cy - u[1] * sgn * 3.6)
                    rr = (r[0] * sgn, r[1] * sgn)
                    self._gadd(3, COL_WHITE, self._rect_poly(sp, (u[0] * sgn, u[1] * sgn), rr, -0.25, 0.25,
                                                             0.15, cw.road_half - 0.15), 130.0)

    # ------------------------------------------------------------- flyovers
    def _build_bridges(self):
        for e in self.fly_edges:
            rt = e.rt
            hw = rt.half
            hd = hw + rt.side_w * 0.6 + 0.3
            p0, p1 = e.cl[0], e.cl[-1]
            L = e.length
            ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
            rx, ry = -uy, ux
            n = int(math.ceil(L / 10.0))
            code = 2 if rt.rank >= 2 else 1

            def open_sides(node):
                if node.kind != 'junction':
                    return set()
                out = set()
                if LEFT[e.d] in node.legs:
                    out.add(-1)
                if RIGHT[e.d] in node.legs:
                    out.add(1)
                return out
            open_a, open_b = open_sides(e.a), open_sides(e.b)
            for k in range(n):
                a, b = L * k / n, L * (k + 1) / n
                c0 = (p0[0] + ux * a, p0[1] + uy * a)
                c1 = (p0[0] + ux * b, p0[1] + uy * b)
                z0, z1 = e.elev.z(a), e.elev.z(b)
                solid = min(z0, z1) < DECK_H * 0.985
                marks = []

                def quad(o0, o1, col, lift=0.03, f0=0.0, f1=1.0):
                    q0 = (c0[0] + (c1[0] - c0[0]) * f0, c0[1] + (c1[1] - c0[1]) * f0, z0 + (z1 - z0) * f0 + lift)
                    q1 = (c0[0] + (c1[0] - c0[0]) * f1, c0[1] + (c1[1] - c0[1]) * f1, z0 + (z1 - z0) * f1 + lift)
                    marks.append(([(q0[0] + rx * o0, q0[1] + ry * o0, q0[2]), (q1[0] + rx * o0, q1[1] + ry * o0, q1[2]),
                                   (q1[0] + rx * o1, q1[1] + ry * o1, q1[2]), (q0[0] + rx * o1, q0[1] + ry * o1, q0[2])], col))
                st = rt.style
                if st == 'divided':
                    quad(-rt.median, rt.median, (112, 110, 106), 0.04)
                    for o in (-(rt.median + 0.12), rt.median + 0.12):
                        quad(o - 0.09, o + 0.09, COL_YELLOW)
                    for o in (-(hw - 0.3), hw - 0.3):
                        quad(o - 0.09, o + 0.09, COL_WHITE)
                    for o in (-4.6, 4.6):
                        quad(o - 0.08, o + 0.08, COL_WHITE, 0.03, 0.1, 0.4)
                elif st == 'avenue':
                    quad(-rt.median, rt.median, COL_MEDIAN, 0.05)
                    for o in (-(hw - 0.25), hw - 0.25):
                        quad(o - 0.09, o + 0.09, COL_WHITE)
                    for o in (-4.05, 4.05):
                        quad(o - 0.08, o + 0.08, COL_WHITE, 0.03, 0.1, 0.4)
                elif st == 'street':
                    for o in (-0.17, 0.17):
                        quad(o - 0.07, o + 0.07, COL_YELLOW)
                elif st == 'oneway':
                    quad(-0.08, 0.08, COL_WHITE, 0.03, 0.1, 0.4)
                    for o in (-(hw - 0.25), hw - 0.25):
                        quad(o - 0.09, o + 0.09, COL_WHITE)
                osd = set()
                if k == 0:
                    osd |= open_a
                if k == n - 1:
                    osd |= open_b
                med_open = bool(osd)
                ch = dict(c0=c0, c1=c1, z0=z0, z1=z1, r=(rx, ry), hw=hw, hd=hd, solid=solid, marks=marks,
                          open_sides=osd, median=(rt.barrier and not med_open), road_col=rt.road_col, side_col=rt.side_col,
                          cx=(c0[0] + c1[0]) * 0.5, cy=(c0[1] + c1[1]) * 0.5, rad=hd + 7.0)
                self.decks.append(ch)
                corners = [(c0[0] + rx * hd, c0[1] + ry * hd), (c0[0] - rx * hd, c0[1] - ry * hd),
                           (c1[0] + rx * hd, c1[1] + ry * hd), (c1[0] - rx * hd, c1[1] - ry * hd)]
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                self.deck_hash.insert(ch, min(xs), min(ys), max(xs), max(ys))
                lvl_s = 2 if solid or min(z0, z1) < 3.2 else 1
                poly = [(c0[0] + rx * hw, c0[1] + ry * hw), (c1[0] + rx * hw, c1[1] + ry * hw),
                        (c1[0] - rx * hw, c1[1] - ry * hw), (c0[0] - rx * hw, c0[1] - ry * hw)]
                self._surf(poly, code, rt.limit_kmh, lvl_s)
                # collision parapets (invisible: drawn as part of the deck)
                zm = (z0 + z1) * 0.5
                lv = 2 if (solid or zm < 3.2) else 1
                for side in (-1, 1):
                    if side not in osd:
                        self._piece(c0, c1, rx, ry, side * (hd - 0.1), 0.16, lv)
                if rt.barrier and not med_open:
                    self._piece(c0, c1, rx, ry, 0.0, 0.42, lv)
            # pillars
            uh = 0.0
            u = RAMP_LEN + 12.0
            while u < L - RAMP_LEN - 10.0:
                skip = False
                for fo in self.flyovers:
                    if fo['over'] is e and abs(u - fo['u']) < fo['under_rt'].half + fo['under_rt'].side_w + 6.0:
                        skip = True
                if not skip:
                    px, py = p0[0] + ux * u, p0[1] + uy * u
                    sz = 1.0 if rt.rank >= 2 else 0.7
                    self.pillars.append(dict(x=px, y=py, s=sz, ux=ux, uy=uy, h=DECK_H - DECK_T))
                    self.circles.append((px, py, sz * 1.15, 'pillar', DECK_H, 0))
                u += 26.0
            # lamps on the deck
            u = RAMP_LEN * 0.5
            while u < L - RAMP_LEN * 0.5:
                z = e.elev.z(u)
                for side in (-1, 1):
                    px, py = p0[0] + ux * u + rx * side * (hd - 0.55), p0[1] + uy * u + ry * side * (hd - 0.55)
                    self.lamps.append((px, py, 7.0, side, z))
                    self.circles.append((px, py, 0.2, 'lamp', 7.0 + z, 2 if z < 3.2 else 1))
                u += 42.0

    def _piece(self, c0, c1, rx, ry, lat, ht, lvl):
        ax, ay = c0[0] + rx * lat, c0[1] + ry * lat
        bx, by = c1[0] + rx * lat, c1[1] + ry * lat
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln * ht, dx / ln * ht
        ex, ey = dx / ln * 0.05, dy / ln * 0.05
        poly = [(ax - ex + nx, ay - ey + ny), (bx + ex + nx, by + ey + ny),
                (bx + ex - nx, by + ey - ny), (ax - ex - nx, ay - ey - ny)]
        self.barriers.append((ax, ay, bx, by, ht, 0.9, 'median' if ht > 0.3 else 'rail', poly, lvl, False))

    # ---------------------------------------------------------------- barriers
    def _add_barrier_line(self, pts, half_thick, height, kind, piece=20.0):
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            L = math.hypot(x2 - x1, y2 - y1)
            n = max(1, int(math.ceil(L / piece)))
            for k in range(n):
                ax, ay = x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n
                bx, by = x1 + (x2 - x1) * (k + 1) / n, y1 + (y2 - y1) * (k + 1) / n
                dx, dy = bx - ax, by - ay
                ln = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / ln * half_thick, dx / ln * half_thick
                ex, ey = dx / ln * 0.05, dy / ln * 0.05
                poly = [(ax - ex + nx, ay - ey + ny), (bx + ex + nx, by + ey + ny),
                        (bx + ex - nx, by + ey - ny), (ax - ex - nx, ay - ey - ny)]
                self.barriers.append((ax, ay, bx, by, half_thick, height, kind, poly, 0, True))

    def _build_barriers(self):
        C = Config
        for e in self.edges:
            rt = e.rt
            if not rt.barrier or e.elev is not None:
                continue
            L = e.length
            lead = C.LEAD + 1.5

            def side_open(node, direction):
                return node.kind == 'junction' and direction in node.legs
            m0 = lead if (side_open(e.a, RIGHT[e.d]) or side_open(e.a, LEFT[e.d])) else 0.0
            m1 = lead if (side_open(e.b, RIGHT[e.d]) or side_open(e.b, LEFT[e.d])) else 0.0
            if L - m0 - m1 > 2.0:
                self._add_barrier_line(pl_slice(e.cl, e.cum, m0, L - m1), 0.42, 0.95, 'median')
            off = rt.half + 0.75
            for side, dirn in ((-1, LEFT[e.d]), (1, RIGHT[e.d])):
                t0 = (lead + 3.5) if side_open(e.a, dirn) else 0.0
                t1 = (lead + 3.5) if side_open(e.b, dirn) else 0.0
                if L - t0 - t1 > 2.0:
                    sub = pl_slice(e.cl, e.cum, t0, L - t1)
                    self._add_barrier_line(pl_offset(sub, off * side), 0.14, 0.75, 'rail')
        for n in self.nodes.values():
            rk, ck = n.row_rt, n.col_rt
            if n.kind == 'bend' and (rk.barrier or ck.barrier):
                legs = list(n.legs)
                t_in, t_out = OPP[legs[0]], legs[1]
                self._add_barrier_line(self.bend_curve(n, t_in, t_out, 0.0, 26), 0.42, 0.95, 'median', 8.0)
                hw = max(n.legs[legs[0]].rt.half, n.legs[legs[1]].rt.half)
                for o in (-(hw + 0.75), hw + 0.75):
                    self._add_barrier_line(self.bend_curve(n, t_in, t_out, o, 26), 0.14, 0.75, 'rail', 8.0)
            elif n.kind == 'junction' and (rk.barrier or ck.barrier):
                if rk.barrier:
                    off = rk.half + 0.75
                    for d, sy in (('N', -1), ('S', 1)):
                        if d not in n.legs:
                            self._add_barrier_line([(n.x - n.ext['W'], n.y + sy * off),
                                                    (n.x + n.ext['E'], n.y + sy * off)], 0.14, 0.75, 'rail')
                if ck.barrier:
                    off = ck.half + 0.75
                    for d, sx in (('W', -1), ('E', 1)):
                        if d not in n.legs:
                            self._add_barrier_line([(n.x + sx * off, n.y - n.ext['N']),
                                                    (n.x + sx * off, n.y + n.ext['S'])], 0.14, 0.75, 'rail')

    # --------------------------------------------------------------- furniture
    def _build_furniture(self):
        C = Config
        rng = random.Random(self.seed * 17 + 3)
        for e in self.edges:
            if e.elev is not None:
                continue
            rt = e.rt
            L = e.length
            mid = e.midblock
            fk = rt.furniture
            hw = rt.half

            # Roadside Speed Limit Signboards along edges (Right sidewalk curb)
            if L > 40.0:
                s_sign = min(22.0, L * 0.35)
                limit_val = 80 if fk == 'highway' else (60 if fk == 'avenue' else 40)
                try:
                    sx_p, sy_p, sh_p = e.pose(s_sign, -(hw + 1.4))
                    self.signs.append({
                        'x': sx_p, 'y': sy_p, 'z': 0.0, 'h': 2.7,
                        'fx': -math.cos(sh_p), 'fy': -math.sin(sh_p),
                        'kind': 'speed', 'val': limit_val
                    })
                except Exception:
                    pass

            def near_mid(s, w):
                return mid is not None and abs(s - L * 0.5) < w

            def under_deck(x, y):
                return self.deck_z(x, y) is not None or bool(self.deck_hash.query_box(x - 6.0, y - 6.0, x + 6.0, y + 6.0))

            if fk == 'highway':
                s = 24.0
                while s < L - 24.0:
                    for side in (-1, 1):
                        x, y, _ = e.pose(s, side * (hw + 2.6))
                        if not under_deck(x, y):
                            self.lamps.append((x, y, 9.0, side, 0.0))
                            self.circles.append((x, y, 0.22, 'lamp', 9.0, 0))
                    s += 52.0
                continue
            if fk == 'avenue':
                s = 22.0
                while s < L - 20.0:
                    if not near_mid(s, 6):
                        x, y, _ = e.pose(s, 0.0)
                        if not under_deck(x, y):
                            self.trees.append((x, y, rng.uniform(5.5, 7.5), rng.random()))
                            self.circles.append((x, y, 0.35, 'tree', 6.0, 0))
                    s += 15.0
                s = 30.0
                idx = 0
                while s < L - 24.0:
                    for side in (-1, 1):
                        x, y, _ = e.pose(s, side * (hw + 0.9))
                        if not under_deck(x, y):
                            self.lamps.append((x, y, 8.0, side, 0.0))
                            self.circles.append((x, y, 0.2, 'lamp', 8.0, 0))
                    s += 38.0
                continue
            gap, lgap = (24.0, 60.0) if fk == 'lane' else (17.0, 40.0)
            idx = 0
            s = 14.0
            while s < L - 13.0:
                if not near_mid(s, 7):
                    side = 1 if idx % 2 == 0 else -1
                    x, y, _ = e.pose(s, side * (hw + (1.6 if fk == 'lane' else 1.9)))
                    if not under_deck(x, y):
                        self.trees.append((x, y, rng.uniform(5.0, 8.5), rng.random()))
                        self.circles.append((x, y, 0.35, 'tree', 6.0, 0))
                idx += 1
                s += gap
            s = 26.0
            idx = 0
            while s < L - 16.0:
                if not near_mid(s, 8):
                    side = -1 if idx % 2 == 0 else 1
                    x, y, _ = e.pose(s, side * (hw + 0.9))
                    if not under_deck(x, y):
                        self.lamps.append((x, y, 7.2, side, 0.0))
                        self.circles.append((x, y, 0.2, 'lamp', 7.2, 0))
                idx += 1
                s += lgap

    # --------------------------------------------------------------- buildings
    def _forbid_rects(self):
        out = []
        for e in self.edges:
            rt = e.rt
            m = rt.half + rt.side_w + (4.8 if rt.rank >= 3 else 1.3)
            if e.elev is not None:
                m += 16.0  # Wide clear corridor so flyovers never clip into buildings
            cl, cum = e.cl, e.cum
            L = e.length
            step = 12.0
            k = 0
            while k * step < L:
                sub = pl_slice(cl, cum, k * step, min(L, (k + 1) * step + 0.5))
                xs = [p[0] for p in sub]
                ys = [p[1] for p in sub]
                out.append((min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m))
                k += 1
        for ch in self.decks:
            cx, cy, rad = ch['cx'], ch['cy'], ch['rad']
            out.append((cx - rad - 14.0, cy - rad - 14.0, cx + rad + 14.0, cy + rad + 14.0))
        for n in self.nodes.values():
            rk, ck = n.row_rt, n.col_rt
            M = max(rk.side_w + (4.8 if rk.rank >= 3 else 1.3), ck.side_w + (4.8 if ck.rank >= 3 else 1.3))
            if n.kind == 'junction':
                out.append((n.x - n.ex - M, n.y - n.ey - M, n.x + n.ex + M, n.y + n.ey + M))
            else:
                d = n.ext[list(n.legs)[0]]
                out.append((n.x - d - M, n.y - d - M, n.x + d + M, n.y + d + M))
        return out

    def _build_buildings(self):
        C = Config
        rng = random.Random(self.seed * 23 + 11)
        W = C.WALL_OFFSET
        X0, X1, Y0, Y1 = min(self.xs), max(self.xs), min(self.ys), max(self.ys)
        lo_x, hi_x, lo_y, hi_y = X0 - W + 10.0, X1 + W - 10.0, Y0 - W + 10.0, Y1 + W - 10.0
        self.bounds = (X0 - W - 5.0, Y0 - W - 5.0, X1 + W + 5.0, Y1 + W + 5.0)
        forbid = SpatialHash(80.0)
        for f in self._forbid_rects():
            forbid.insert(f, *f)
        placed = SpatialHash(80.0)
        bid = [0]
        cxm0, cym0 = (X0 + X1) * 0.5, (Y0 + Y1) * 0.5
        span = max(X1 - X0, Y1 - Y0)

        def free(x0, y0, x1, y1, gap=2.0):
            if x0 < lo_x or y0 < lo_y or x1 > hi_x or y1 > hi_y:
                return False
            for f in forbid.query_box(x0, y0, x1, y1):
                if x0 < f[2] and x1 > f[0] and y0 < f[3] and y1 > f[1]:
                    return False
            for b in placed.query_box(x0 - gap, y0 - gap, x1 + gap, y1 + gap):
                if x0 - gap < b.x1 and x1 + gap > b.x0 and y0 - gap < b.y1 and y1 + gap > b.y0:
                    return False
            return True

        def add(x0, y0, x1, y1):
            dist = math.hypot((x0 + x1) * 0.5 - cxm0, (y0 + y1) * 0.5 - cym0)
            h = rng.uniform(9.0, 20.0) + max(0.0, 1.0 - dist / (span * 0.5)) * rng.uniform(0.0, 58.0)
            b = Building(x0, y0, x1, y1, h, rng.choice(BUILDING_COLORS), rng.randint(0, 2), bid[0])
            bid[0] += 1
            self.buildings.append(b)
            placed.insert(b, x0, y0, x1, y1)

        self.destinations = []
        lm_specs = [
            ('airport', 'AIRPORT', 70.0, 34.0, 15.0, (215, 222, 230), 'AIRPORT', (0, 210, 255), 'International Airport & Runway', ['H', 'A']),
            ('hospital', 'HOSPITAL', 40.0, 30.0, 26.0, (238, 240, 246), 'HOSPITAL', (255, 50, 50), 'City Medical Center & Helipad', ['A', 'S']),
            ('mall', 'SHOPPING MALL', 44.0, 32.0, 18.0, (190, 175, 155), 'MALL', (255, 200, 40), 'Grand City Mall & Atrium', ['A', 'S', 'H']),
            ('skyscraper', 'SKYSCRAPER TOWER', 32.0, 30.0, 96.0, (85, 130, 180), 'TOWER', (100, 180, 255), 'Downtown Skyline High-Rise (96m)', ['A', 'H', 'S']),
            ('market', 'CENTRAL MARKET', 36.0, 28.0, 12.0, (168, 148, 130), 'MARKET', (255, 140, 30), 'City Bazaar & Vendor Stalls', ['S', 'L', 'A']),
            ('bus_station', 'BUS TERMINUS', 42.0, 30.0, 14.0, (150, 155, 165), 'BUS', (40, 210, 100), 'Central City Bus Terminal', ['S', 'A', 'L']),
            ('railway', 'RAILWAY STATION', 48.0, 32.0, 18.0, (165, 80, 65), 'TRAIN', (230, 90, 240), 'Grand Central Train Station & Tracks', ['A', 'S', 'H'])
        ]
        used_edges = set()
        lm_rng = random.Random(self.seed * 53 + 7)

        for tag, name, bw, bd, bh, col, icon, color, desc, pref_rts in lm_specs:
            found = False
            # Find candidate edges with ample length and no flyovers
            cands = [e for e in self.edges if e.elev is None and e.rt.key in pref_rts and e.length >= bw + 20.0 and id(e) not in used_edges]
            if not cands:
                cands = [e for e in self.edges if e.elev is None and e.length >= bw + 14.0 and id(e) not in used_edges]
            if not cands:
                cands = [e for e in self.edges if e.elev is None and id(e) not in used_edges]

            lm_rng.shuffle(cands)

            for e in cands:
                rt = e.rt
                hw, sw = rt.half, rt.side_w
                dw_len = 28.0
                dw_w = 20.0

                s = e.length * 0.5
                for side in (1, -1):
                    lane_key = e.d if side == 1 else OPP.get(e.d, e.d)
                    lanes = e.lanes.get(lane_key, [])
                    if not lanes:
                        continue
                    app_lane = lanes[-1] if side == 1 else lanes[0]

                    curb_x, curb_y, h = e.pose(s, side * (hw + sw))
                    fx, fy = math.cos(h), math.sin(h)
                    rx, ry = -fy, fx
                    out_x, out_y = rx * side, ry * side

                    cx_b = curb_x + out_x * (dw_len + 12.0 + bd * 0.5)
                    cy_b = curb_y + out_y * (dw_len + 12.0 + bd * 0.5)

                    x0 = min(cx_b - abs(fx) * bw * 0.5 - abs(out_x) * bd * 0.5, cx_b + abs(fx) * bw * 0.5 + abs(out_x) * bd * 0.5)
                    x1 = max(cx_b - abs(fx) * bw * 0.5 - abs(out_x) * bd * 0.5, cx_b + abs(fx) * bw * 0.5 + abs(out_x) * bd * 0.5)
                    y0 = min(cy_b - abs(fy) * bw * 0.5 - abs(out_y) * bd * 0.5, cy_b + abs(fy) * bw * 0.5 + abs(out_y) * bd * 0.5)
                    y1 = max(cy_b - abs(fy) * bw * 0.5 - abs(out_y) * bd * 0.5, cy_b + abs(fy) * bw * 0.5 + abs(out_y) * bd * 0.5)

                    if x0 < lo_x or x1 > hi_x or y0 < lo_y or y1 > hi_y:
                        continue
                    if not free(x0, y0, x1, y1, 2.0):
                        continue

                    # Paved entrance driveway polygon (width 20m, length 28m)
                    dw_poly = [
                        (curb_x - fx * (dw_w * 0.5), curb_y - fy * (dw_w * 0.5)),
                        (curb_x + fx * (dw_w * 0.5), curb_y + fy * (dw_w * 0.5)),
                        (curb_x + fx * (dw_w * 0.5) + out_x * dw_len, curb_y + fy * (dw_w * 0.5) + out_y * dw_len),
                        (curb_x - fx * (dw_w * 0.5) + out_x * dw_len, curb_y - fy * (dw_w * 0.5) + out_y * dw_len)
                    ]

                    entry_pt = (curb_x + out_x * 6.0, curb_y + out_y * 6.0)
                    finish_pt = (curb_x + out_x * 15.0, curb_y + out_y * 15.0)
                    gate_p1 = (entry_pt[0] - fx * 10.0, entry_pt[1] - fy * 10.0)
                    gate_p2 = (entry_pt[0] + fx * 10.0, entry_pt[1] + fy * 10.0)

                    # Mission finish bay box (8m x 10m)
                    fb_w, fb_d = 4.2, 4.8
                    finish_box = [
                        (finish_pt[0] - fx * fb_w - out_x * fb_d, finish_pt[1] - fy * fb_w - out_y * fb_d),
                        (finish_pt[0] + fx * fb_w - out_x * fb_d, finish_pt[1] + fy * fb_w - out_y * fb_d),
                        (finish_pt[0] + fx * fb_w + out_x * fb_d, finish_pt[1] + fy * fb_w + out_y * fb_d),
                        (finish_pt[0] - fx * fb_w + out_x * fb_d, finish_pt[1] - fy * fb_w - out_y * fb_d)
                    ]

                    all_dw_pts = dw_poly + finish_box
                    dw_box = (
                        min(p[0] for p in all_dw_pts) - 10.0, min(p[1] for p in all_dw_pts) - 10.0,
                        max(p[0] for p in all_dw_pts) + 10.0, max(p[1] for p in all_dw_pts) + 10.0
                    )
                    # Ensure no existing building blocks the driveway corridor
                    dw_blocked = False
                    for ob in placed.query_box(*dw_box):
                        if dw_box[0] < ob.x1 and dw_box[2] > ob.x0 and dw_box[1] < ob.y1 and dw_box[3] > ob.y0:
                            dw_blocked = True
                            break
                    if dw_blocked:
                        continue

                    self._gadd(1, (52, 54, 58), dw_poly)
                    self._surf(dw_poly, 1, 35.0, 0)
                    self._gadd(2, (24, 28, 36), finish_box)

                    # Create landmark building
                    b = Building(x0, y0, x1, y1, bh, col, 0, bid[0])
                    b.landmark = dict(
                        tag=tag, name=name, desc=desc, icon=icon, color=color,
                        curb_pt=(curb_x, curb_y), entry_pt=entry_pt, finish_pt=finish_pt,
                        gate_p1=gate_p1, gate_p2=gate_p2, dw_poly=dw_poly, finish_box=finish_box,
                        out_vec=(out_x, out_y), f_vec=(fx, fy), bw=bw, bd=bd, bh=bh,
                        edge=e, side=side, lane=app_lane, s=s, corridor_box=dw_box
                    )
                    bid[0] += 1
                    self.buildings.append(b)
                    placed.insert(b, x0, y0, x1, y1)
                    # Reserve driveway corridor in forbid so NO filler building can ever block the entrance
                    forbid.insert(dw_box, *dw_box)

                    self.destinations.append(dict(
                        tag=tag, name=name, desc=desc, icon=icon, color=color,
                        x=finish_pt[0], y=finish_pt[1], b_center=(cx_b, cy_b), b=b,
                        edge=e, side=side, lane=app_lane, s=s,
                        curb_pt=(curb_x, curb_y), entry_pt=entry_pt, finish_pt=finish_pt,
                        gate_p1=gate_p1, gate_p2=gate_p2, dw_poly=dw_poly, finish_box=finish_box,
                        out_vec=(out_x, out_y), f_vec=(fx, fy), corridor_box=dw_box
                    ))
                    used_edges.add(id(e))
                    found = True
                    break
                if found:
                    break

            # Fallback if edge-adjacent placement didn't find room
            if not found:
                for _ in range(400):
                    x0 = lm_rng.uniform(lo_x, hi_x - bw)
                    y0 = lm_rng.uniform(lo_y, hi_y - bd)
                    if free(x0, y0, x0 + bw, y0 + bd, 3.5):
                        b = Building(x0, y0, x0 + bw, y0 + bd, bh, col, 0, bid[0])
                        b.landmark = dict(tag=tag, name=name, desc=desc, icon=icon, color=color,
                                          curb_pt=((x0+x0+bw)*0.5, (y0+y0+bd)*0.5),
                                          entry_pt=((x0+x0+bw)*0.5, (y0+y0+bd)*0.5),
                                          finish_pt=((x0+x0+bw)*0.5, (y0+y0+bd)*0.5))
                        bid[0] += 1
                        self.buildings.append(b)
                        placed.insert(b, x0, y0, x0 + bw, y0 + bd)
                        self.destinations.append(dict(tag=tag, name=name, desc=desc, icon=icon, color=color,
                                                      x=(x0 + x0 + bw) * 0.5, y=(y0 + y0 + bd) * 0.5, b=b,
                                                      curb_pt=((x0+x0+bw)*0.5, (y0+y0+bd)*0.5),
                                                      entry_pt=((x0+x0+bw)*0.5, (y0+y0+bd)*0.5),
                                                      finish_pt=((x0+x0+bw)*0.5, (y0+y0+bd)*0.5)))
                        break

        for e in self.edges:
            rt = e.rt
            setback = rt.half + rt.side_w + (4.8 if rt.rank >= 3 else 1.5)
            L = e.length
            for side in (-1, 1):
                s = rng.uniform(0.0, 8.0)
                while s < L - 6.0:
                    w = min(rng.uniform(16.0, 34.0), L - s)
                    if w < 10.0:
                        break
                    done = False
                    for depth in (rng.uniform(18.0, 34.0), 22.0, 16.0, 12.0):
                        x, y, h = e.pose(s + w * 0.5, side * (setback + depth * 0.5))
                        horiz = abs(math.cos(h)) >= abs(math.sin(h))
                        hx, hy = (w * 0.5, depth * 0.5) if horiz else (depth * 0.5, w * 0.5)
                        if free(x - hx, y - hy, x + hx, y + hy):
                            add(x - hx, y - hy, x + hx, y + hy)
                            done = True
                            break
                    s += (w + rng.uniform(1.5, 4.0)) if done else 10.0
        for _ in range(9000):
            w = rng.uniform(14.0, 36.0)
            d = rng.uniform(14.0, 36.0)
            x0 = rng.uniform(lo_x, hi_x - w)
            y0 = rng.uniform(lo_y, hi_y - d)
            if free(x0, y0, x0 + w, y0 + d, 3.0):
                add(x0, y0, x0 + w, y0 + d)
        t = 5.0
        bx0, by0, bx1, by1 = X0 - W, Y0 - W, X1 + W, Y1 + W
        walls = [(bx0 - t, by0 - t, bx1 + t, by0), (bx0 - t, by1, bx1 + t, by1 + t),
                 (bx0 - t, by0, bx0, by1), (bx1, by0, bx1 + t, by1)]
        for x0, y0, x1, y1 in walls:
            b = Building(x0, y0, x1, y1, 6.0, (128, 128, 132), 0, bid[0], wall=True)
            bid[0] += 1
            self.buildings.append(b)
        target = len(self.trees) + 320
        tries = 0
        while len(self.trees) < target and tries < 8000:
            tries += 1
            x = rng.uniform(lo_x, hi_x)
            y = rng.uniform(lo_y, hi_y)
            if free(x - 1.5, y - 1.5, x + 1.5, y + 1.5, 1.0):
                self.trees.append((x, y, rng.uniform(5.0, 9.5), rng.random()))
                self.circles.append((x, y, 0.4, 'tree', 6.0, 0))
        # Remove any non-landmark building that overlaps landmark driveway corridor
        lm_blds = {d['b'] for d in self.destinations if 'b' in d}
        def bld_in_corridor(bld):
            if bld in lm_blds:
                return False
            for d in self.destinations:
                bx = d.get('corridor_box')
                if bx and (bld.x0 < bx[2] and bld.x1 > bx[0] and bld.y0 < bx[3] and bld.y1 > bx[1]):
                    return True
            return False
        self.buildings = [bld for bld in self.buildings if not bld_in_corridor(bld)]

        # Ensure zero barriers, trees, circles, or lamps inside landmark driveway corridors
        for d in self.destinations:
            dw = d.get('dw_poly')
            cx, cy = d.get('curb_pt', (0.0, 0.0))
            if dw:
                def bar_in_dw(bar):
                    for px, py in bar[7]:
                        if poly_contains(dw, px, py):
                            return True
                    mx, my = (bar[0] + bar[2]) * 0.5, (bar[1] + bar[3]) * 0.5
                    if poly_contains(dw, mx, my):
                        return True
                    if (mx - cx) ** 2 + (my - cy) ** 2 < 144.0:
                        return True
                    return False

                self.barriers = [b for b in self.barriers if not bar_in_dw(b)]
                self.trees = [t for t in self.trees if not poly_contains(dw, t[0], t[1]) and ((t[0] - cx) ** 2 + (t[1] - cy) ** 2 >= 144.0)]
                self.circles = [c for c in self.circles if not poly_contains(dw, c[0], c[1]) and ((c[0] - cx) ** 2 + (c[1] - cy) ** 2 >= 144.0)]
                self.lamps = [l for l in self.lamps if not poly_contains(dw, l[0], l[1]) and ((l[0] - cx) ** 2 + (l[1] - cy) ** 2 >= 144.0)]

        # Airplane completely removed as requested
        self.airplane = None

    # ----------------------------------------------------------- static index
    def _index_static(self):
        for b in self.buildings:
            self.col_hash.insert(('rect', b, 0), b.x0, b.y0, b.x1, b.y1)
            self.bld_hash.insert(b, b.cx - b.rad, b.cy - b.rad, b.cx + b.rad, b.cy + b.rad)
        for c in self.circles:
            self.col_hash.insert(('circle', c, c[5]), c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2])
        for bar in self.barriers:
            xs = [p[0] for p in bar[7]]
            ys = [p[1] for p in bar[7]]
            self.col_hash.insert(('bar', bar, bar[8]), min(xs), min(ys), max(xs), max(ys))
            if bar[9]:
                self.prop_hash.insert(('bar', bar), min(xs), min(ys), max(xs), max(ys))
        for t in self.trees:
            self.prop_hash.insert(('tree', t), t[0] - 4, t[1] - 4, t[0] + 4, t[1] + 4)
        for l in self.lamps:
            self.prop_hash.insert(('lamp', l), l[0] - 2, l[1] - 2, l[0] + 2, l[1] + 2)
        for h in self.signal_heads:
            self.prop_hash.insert(('sig', h), h['x'] - 2, h['y'] - 2, h['x'] + 2, h['y'] + 2)
        for p in self.pillars:
            self.prop_hash.insert(('pillar', p), p['x'] - 3, p['y'] - 3, p['x'] + 3, p['y'] + 3)
        for s in getattr(self, 'signs', []):
            self.prop_hash.insert(('sign', s), s['x'] - 2, s['y'] - 2, s['x'] + 2, s['y'] + 2)

    # ----------------------------------------------------------------- queries
    def surface_at(self, x, y, lvl=0):
        """-> (code, limit_kmh): code 0 grass, 1 street, 2 highway/avenue, 3 sidewalk."""
        best = (0, 0)
        for poly, code, limit, il in self.surf_hash.query_box(x, y, x, y):
            if il != 2 and il != lvl:
                continue
            if poly_contains(poly, x, y):
                if code in (1, 2):
                    return (code, limit)
                best = (3, 0)
        return best

    def static_collision(self, corners, lvl=0):
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        for kind, obj, il in self.col_hash.query_box(min(xs), min(ys), max(xs), max(ys)):
            if il != 2 and il != lvl:
                continue
            if kind == 'rect':
                if sat_polygon_collision(corners, obj.poly):
                    return "HIT THE CITY WALL!" if obj.wall else "HIT A BUILDING!"
            elif kind == 'circle':
                if self._circle_poly(obj[0], obj[1], obj[2], corners):
                    return {"tree": "HIT A TREE!", "lamp": "HIT A LAMP POST!", "pole": "HIT A SIGNAL POLE!",
                            "pillar": "HIT A BRIDGE PILLAR!"}.get(obj[3], "HIT AN OBSTACLE!")
            else:
                if sat_polygon_collision(corners, obj[7]):
                    return "HIT THE MEDIAN BARRIER!" if obj[6] == 'median' else "HIT THE GUARDRAIL!"
        return ""

    @staticmethod
    def _circle_poly(px, py, pr, poly):
        n = len(poly)
        if poly_contains(poly, px, py):
            return True
        for i in range(n):
            d, _ = seg_point_dist(px, py, poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1])
            if d < pr:
                return True
        return False

    def deck_z(self, x, y, on_deck_hint=False):
        """Elevation of a flyover surface at (x, y) or None if not over a deck chunk."""
        for ch in self.deck_hash.query_box(x, y, x, y):
            ax, ay = ch['c0']
            bx, by = ch['c1']
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = ((x - ax) * dx + (y - ay) * dy) / l2
            if -0.001 <= t <= 1.001:
                rx, ry = ch['r']
                lat = (x - ax) * rx + (y - ay) * ry
                if abs(lat) <= ch['hd']:
                    return ch['z0'] + (ch['z1'] - ch['z0']) * clamp(t, 0.0, 1.0)
        return None

    # -------------------------------------------------------------- navigation
    def _build_nav(self):
        self.path_hash = SpatialHash(40.0)
        paths = list(self.lanes)
        for n in self.nodes.values():
            paths.extend(n.conns)
        self.all_paths = paths
        self._find_main_network(paths)
        for p in paths:
            for i in range(len(p.pts) - 1):
                a, b = p.pts[i], p.pts[i + 1]
                self.path_hash.insert((p, i), min(a[0], b[0]) - 1, min(a[1], b[1]) - 1,
                                      max(a[0], b[0]) + 1, max(a[1], b[1]) + 1)

    def _find_main_network(self, paths):
        """Largest strongly connected set of lanes/connectors: everything else is a trap and is never used."""
        adj = {p.pid: [q.pid for q in (p.out if p.kind == 'lane' else [p.out_lane]) if q is not None] for p in paths}
        radj = defaultdict(list)
        for u, vs in adj.items():
            for v in vs:
                radj[v].append(u)
        order, seen = [], set()
        for s0 in adj:
            if s0 in seen:
                continue
            st = [(s0, iter(adj[s0]))]
            seen.add(s0)
            while st:
                u, it = st[-1]
                adv = False
                for v in it:
                    if v not in seen:
                        seen.add(v)
                        st.append((v, iter(adj[v])))
                        adv = True
                        break
                if not adv:
                    order.append(u)
                    st.pop()
        comp, best = {}, []
        for s0 in reversed(order):
            if s0 in comp:
                continue
            members = []
            st = [s0]
            comp[s0] = s0
            while st:
                u = st.pop()
                members.append(u)
                for v in radj[u]:
                    if v not in comp:
                        comp[v] = s0
                        st.append(v)
            if len(members) > len(best):
                best = members
        main = set(best)
        for p in paths:
            p.main = p.pid in main
        self.main_lanes = [l for l in self.lanes if l.main]
        self.main_frac = len(main) / max(1, len(paths))

    def nearest_path(self, x, y, heading, z=0.0, maxd=6.0):
        """-> (path, s, dist) of the drivable path best matching a pose, or None."""
        best = None
        bd = 1e9
        for (p, i) in self.path_hash.query_box(x - maxd, y - maxd, x + maxd, y + maxd):
            a, b = p.pts[i], p.pts[i + 1]
            d, t = seg_point_dist(x, y, a[0], a[1], b[0], b[1])
            if d > maxd:
                continue
            if math.cos(normalize_angle(p.hd[i] - heading)) < 0.35:
                continue
            s = p.cum[i] + (p.cum[i + 1] - p.cum[i]) * t
            if p.elev is not None and abs(p.zat(s) - z) > 3.5:
                continue
            score = d + (0.0 if p.kind == 'lane' else 0.3)
            if score < bd:
                bd = score
                best = (p, s, d)
        return best

    def plan_route(self, start_path, goal_lane, min_rmin=0.0):
        """Dijkstra over lane/connector graph (travel-time cost). Returns list of paths or None."""
        import heapq
        if start_path is goal_lane:
            return [start_path]
        dist = {start_path.pid: 0.0}
        prev = {}
        pq = [(0.0, start_path.pid, start_path)]
        seen = set()
        while pq:
            d, pid, p = heapq.heappop(pq)
            if pid in seen:
                continue
            seen.add(pid)
            if p is goal_lane:
                route = [p]
                while route[-1].pid in prev:
                    route.append(prev[route[-1].pid])
                return route[::-1]
            nxts = p.out if p.kind == 'lane' else [p.out_lane]
            for q in nxts:
                if q is None:
                    continue
                if min_rmin and q.kind == 'conn' and q.move != 'S' and q.rmin < min_rmin:
                    continue
                nd = d + q.length / max(q.limit * (0.6 if q.kind == 'conn' else 1.0), 2.0)
                if nd < dist.get(q.pid, 1e18):
                    dist[q.pid] = nd
                    prev[q.pid] = p
                    heapq.heappush(pq, (nd, q.pid, q))
        return None

    def astar_route(self, start_path, goal_lane, goal_pos, min_rmin=0.0):
        """
        A* search algorithm on directed lane/connector network.
        Finds the shortest distance route ("smallest path") from start_path to goal_lane.
        - g(n): Accumulated physical distance (metres) along lanes and connectors.
        - h(n): Admissible Euclidean heuristic = straight-line distance to goal.
        """
        import heapq
        if start_path is None or goal_lane is None:
            return None
        if start_path is goal_lane:
            return [start_path]

        def h(p):
            ex, ey = p.pts[-1][0], p.pts[-1][1]
            return math.hypot(ex - goal_pos[0], ey - goal_pos[1])

        start_pid = start_path.pid
        g_cost = {start_pid: 0.0}
        f_cost = {start_pid: h(start_path)}
        came_from = {}
        pq = [(f_cost[start_pid], 0, start_pid, start_path)]
        seen = set()
        count = 0

        while pq:
            f, _, pid, cur = heapq.heappop(pq)
            if pid in seen:
                continue
            seen.add(pid)

            if cur is goal_lane or (cur.kind == 'conn' and cur.out_lane is goal_lane):
                route = [cur]
                if cur is not goal_lane and cur.kind == 'conn':
                    route.append(cur.out_lane)
                while route[0].pid in came_from:
                    route.insert(0, came_from[route[0].pid])
                return route

            nxts = cur.out if cur.kind == 'lane' else ([cur.out_lane] if cur.out_lane else [])
            for q in nxts:
                if q is None:
                    continue
                if min_rmin and q.kind == 'conn' and q.move != 'S' and q.rmin < min_rmin:
                    continue
                # Physical distance in metres:
                # Add tiny tiebreaker (0.3m) for Left/U turns so straight path is preferred when distances are equal
                step_cost = q.length + (0.3 if (q.kind == 'conn' and q.move in ('L', 'U')) else 0.0)
                tentative_g = g_cost[pid] + step_cost
                if tentative_g < g_cost.get(q.pid, float('inf')):
                    came_from[q.pid] = cur
                    g_cost[q.pid] = tentative_g
                    f_val = tentative_g + h(q)
                    f_cost[q.pid] = f_val
                    count += 1
                    heapq.heappush(pq, (f_val, count, q.pid, q))
        return None

    # ----------------------------------------------------------------- spawns
    def _build_spawns(self):
        rng = random.Random(self.seed * 5 + 1)
        specs = [("HIGHWAY", lambda l: l.edge.rt.key == 'H' and l.edge.elev is None and l.idx == 0),
                 ("FLYOVER  (on the bridge)", lambda l: l.edge.elev is not None and l.idx == 0),
                 ("AVENUE", lambda l: l.edge.rt.key == 'A' and l.edge.elev is None and l.idx == 0),
                 ("CITY STREET", lambda l: l.edge.rt.key == 'S' and l.edge.elev is None),
                 ("NARROW LANE", lambda l: l.edge.rt.key == 'L'),
                 ("ONE-WAY STREET", lambda l: l.edge.rt.key == 'O' and l.idx == 0)]
        for label, pred in specs:
            cands = [l for l in self.main_lanes if pred(l) and l.length > 70.0]
            if not cands:
                continue
            lane = rng.choice(cands)
            self.spawns.append(dict(lane=lane, s=lane.length * 0.5, label=label))
        if not self.spawns:
            lane = self.main_lanes[0]
            self.spawns.append(dict(lane=lane, s=lane.length * 0.5, label="START"))
        # under-the-flyover spawn
        if self.flyovers:
            fo = self.flyovers[0]
            ue = None
            for e in self.edges:
                if e.rt is fo['under_rt'] and e.elev is None:
                    lo = min(e.p0[0], e.p1[0]) - 1, min(e.p0[1], e.p1[1]) - 1
                    hi = max(e.p0[0], e.p1[0]) + 1, max(e.p0[1], e.p1[1]) + 1
                    if lo[0] <= fo['x'] <= hi[0] and lo[1] <= fo['y'] <= hi[1]:
                        ue = e
            if ue is not None:
                t = ue.d
                lane = ue.lanes[t][0]
                # drive-through position: just before the bridge
                best_s = 0.0
                bd = 1e9
                for k in range(int(lane.length)):
                    x, y, _ = lane.pose(float(k))
                    d = math.hypot(x - fo['x'], y - fo['y'])
                    if d < bd:
                        bd, best_s = d, float(k)
                self.spawns.insert(2, dict(lane=lane, s=max(10.0, best_s - 55.0), label="UNDER THE FLYOVER"))
        # Add spawns right outside each of the 7 landmark destinations
        for dest in getattr(self, 'destinations', []):
            bd = 1e9
            best_lane = None
            best_s = 0.0
            for l in self.main_lanes:
                if l.edge.elev is not None:
                    continue
                for k in range(10, max(11, int(l.length) - 10), 4):
                    px, py, _ = l.pose(float(k))
                    d = math.hypot(px - dest['x'], py - dest['y'])
                    if d < bd:
                        bd = d
                        best_lane = l
                        best_s = float(k)
            if best_lane is not None:
                dest['lane'] = best_lane
                dest['s'] = best_s
                self.spawns.append(dict(lane=best_lane, s=best_s, label=f"VISIT: {dest['name']}"))



PLAYER_PAINT = {'sedan': ((0, 200, 255), (30, 30, 36)), 'suv': ((240, 130, 30), (30, 30, 36)),
                'auto': ((240, 200, 20), (30, 30, 34)), 'bike': ((210, 40, 40), (30, 30, 36)),
                'scooter': ((236, 236, 240), (30, 30, 36))}


# =============================================================================
# PLAYER CAR  (metres / seconds / radians)  -  smooth pedal, calm steering
# =============================================================================
class Car:
    """
    Kinematic bicycle model with:
      * progressive pedal  (a tap = gentle, hold = more power)
      * soft speed limiter (the Max Speed slider; no overshoot)
      * cruise control     (holds a set speed, +/- changes it)
      * virtual steering wheel (input ramps, returns to centre, wheel angle is
        limited by lateral acceleration so it is never twitchy at speed)
    """
    kind = 'player'
    id = -1
    prio = 3

    z = 0.0
    lvl = 0

    def __init__(self, vehicle='sedan'):
        self.x = self.y = self.heading = 0.0
        self.v = 0.0
        self.set_vehicle(vehicle)
        self.reset(0.0, 0.0, 0.0)

    def set_vehicle(self, name):
        sp = VEHICLES[name]
        self.vname = name
        self.spec = sp
        self.body = name
        self.length, self.width, self.wheelbase = sp['length'], sp['width'], sp['wheelbase']
        self.max_steer, self.accel = sp['max_steer'], sp['accel']
        self.brake_decel, self.vmax_veh = sp['brake'], sp['vmax']
        self.paint, self.paint2 = PLAYER_PAINT[name]
        self.color, self.color2 = self.paint, self.paint2
        self.rider = (210, 60, 50)

    def update_z(self, city):
        zd = city.deck_z(self.x, self.y)
        if self.on_bridge:
            if zd is None:
                self.on_bridge = False
                self.z = 0.0
            else:
                self.z = zd
        else:
            # Seamless ramp climbing attachment (attaches as soon as vehicle enters incline)
            if zd is not None and (zd < 2.5 or abs(zd - self.z) < 2.0):
                self.on_bridge = True
                self.z = zd
            else:
                self.z = 0.0
        self.lvl = 1 if self.z > 3.2 else 0

    def reset(self, x, y, heading, speed=0.0):
        self.x, self.y, self.heading = x, y, heading
        self.v = speed
        self.z = 0.0
        self.lvl = 0
        self.on_bridge = False
        self.steer_in = 0.0         # virtual steering wheel  [-1, 1]
        self.steer_angle = 0.0      # road wheel angle (rad)
        self.throttle = 0.0         # smoothed pedal
        self.brake = 0.0
        self.cruise_on = False
        self.cruise_v = max(speed, 8.0)
        self.cruise_i = 0.0
        self.stop_hold = 0.0
        self.yaw_rate = 0.0
        self.lat_acc = 0.0
        self.long_acc = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.tire_grip = 1.0
        self.norm_steer = 0.0
        self.norm_throttle = 0.0

    @property
    def speed(self):
        return self.v

    def corners(self):
        return obb_corners(self.x, self.y, self.heading, self.length, self.width)

    def update(self, steer_cmd, throttle_cmd, brake_cmd, dt, city=None):
        C = Config
        v = self.v
        av = abs(v)

        # ---------------- steering wheel (virtual) ----------------
        sens = C.STEER_SENSITIVITY
        rate = C.STEER_RATE * sens * (1.0 / (1.0 + 0.045 * av))        # slower hands at speed
        if abs(steer_cmd) > 0.02:
            tgt = clamp(steer_cmd, -1.0, 1.0)
            # quick when reversing direction, gentle when going further out
            r = rate * (1.8 if (tgt * self.steer_in < 0) else 1.0)
            d = tgt - self.steer_in
            step = r * dt
            self.steer_in += clamp(d, -step, step)
        else:
            r = C.STEER_RETURN_RATE * (1.0 + 0.04 * av) * dt
            if abs(self.steer_in) <= r:
                self.steer_in = 0.0
            else:
                self.steer_in -= math.copysign(r, self.steer_in)

        # wheel angle limited by max lateral acceleration at this speed
        v_ref = max(av, 2.0)
        lat_lim = math.atan(self.wheelbase * C.MAX_LAT_ACCEL / (v_ref * v_ref))
        max_ang = min(self.max_steer, lat_lim) * clamp(0.65 + 0.35 * sens, 0.5, 1.4)
        # response curve: fine control around centre
        shaped = math.copysign(abs(self.steer_in) ** 1.35, self.steer_in)
        target_ang = shaped * max_ang
        self.steer_angle += clamp(target_ang - self.steer_angle, -3.5 * dt, 3.5 * dt)

        # ---------------- pedals ----------------
        thr_target = clamp(throttle_cmd, 0.0, 1.0)
        brk_target = clamp(brake_cmd, 0.0, 1.0)

        if self.cruise_on:
            if brk_target > 0.05:
                self.cruise_on = False
            else:
                err = self.cruise_v - v
                self.cruise_i = clamp(self.cruise_i + err * 0.35 * dt, 0.0, 0.9)
                auto = clamp(0.5 * err + self.cruise_i, 0.0, 1.0)
                thr_target = max(thr_target, auto)
                if err < -0.6:
                    brk_target = max(brk_target, clamp(-0.12 * err, 0.0, 0.25))

        rise = C.PEDAL_RISE * (1.0 if thr_target > self.throttle else 1.0)
        if thr_target > self.throttle:
            self.throttle = min(thr_target, self.throttle + rise * dt)
        else:
            self.throttle = max(thr_target, self.throttle - C.PEDAL_FALL * dt)
        if brk_target > self.brake:
            self.brake = min(brk_target, self.brake + C.BRAKE_RISE * dt)
        else:
            self.brake = max(brk_target, self.brake - 7.0 * dt)

        # ---------------- longitudinal dynamics ----------------
        vmax = min(C.MAX_SPEED, self.vmax_veh)
        a = 0.0

        can_reverse = getattr(self, 'allow_reverse', True)
        in_rev = getattr(self, 'in_reverse', False)

        # Reverse gear engagement logic (responsive, smooth, non-jerky)
        if can_reverse:
            if in_rev:
                # Exit reverse gear if user presses forward throttle (W)
                # or if brake released and vehicle is stationary
                if thr_target > 0.05:
                    self.in_reverse = False
                    in_rev = False
                    self.stop_hold = 0.0
                elif brk_target < 0.02 and v >= -0.05:
                    self.in_reverse = False
                    in_rev = False
            else:
                # Enter reverse quickly after holding brake at full stop (>0.12s)
                if abs(v) < 0.35 and brk_target > 0.05 and thr_target < 0.02:
                    self.stop_hold += dt
                    if self.stop_hold > 0.12:
                        self.in_reverse = True
                        in_rev = True
                else:
                    self.stop_hold = 0.0
        else:
            self.in_reverse = False
            in_rev = False
            self.stop_hold = 0.0

        if in_rev:
            # Responsive, progressive reverse acceleration with pedal
            rev_accel = min(self.accel * 0.75, 3.6)
            taper = clamp((C.MAX_REVERSE_SPEED - abs(v)) / 1.5, 0.0, 1.0) if v < -2.0 else 1.0
            a = -rev_accel * max(0.4, self.brake) * taper
        elif self.throttle > 0.01 and brk_target < 0.05:
            if v < -0.15:
                # Forward pedal while rolling backwards acts as brake
                a = self.brake_decel * 0.75 * self.throttle
            else:
                taper = clamp((vmax - v) / 1.0, 0.0, 1.0)
                power = self.accel * (1.0 - 0.45 * clamp(v / max(vmax, 1.0), 0.0, 1.0))
                a = self.throttle * power * taper
        elif self.brake > 0.01 and not in_rev:
            if v > 0.1:
                a -= self.brake * self.brake_decel
            elif v < -0.1:
                a += self.brake * self.brake_decel
            else:
                a = 0.0

        # Resistive forces (engine braking + rolling resistance)
        if self.throttle < 0.02 and not in_rev:
            if v > 0.05:
                a -= C.ENGINE_BRAKING
            elif v < -0.05:
                a += C.ENGINE_BRAKING
        elif in_rev and self.brake < 0.05:
            if v < -0.05:
                a += C.ENGINE_BRAKING

        a -= math.copysign(C.ROLL_RESISTANCE + C.AIR_DRAG * v * v, v) if abs(v) > 0.02 else 0.0
        if v > vmax:
            a -= 2.5

        nv = v + a * dt

        # Smooth, clean transitions without jitter
        if not in_rev:
            if v >= 0.0 and nv < 0.0:
                nv = 0.0
            elif v <= 0.0 and abs(nv) < 0.05 and thr_target < 0.02:
                nv = 0.0
            elif v <= 0.0 and brk_target > 0.02 and thr_target < 0.02 and not can_reverse:
                nv = 0.0
        else:
            if v < 0.0 and nv > 0.0:
                nv = 0.0

        if abs(nv) < 0.02 and abs(a) < 0.4 and self.throttle < 0.02 and (not in_rev or self.brake < 0.05):
            nv = 0.0

        self.v = clamp(nv, -C.MAX_REVERSE_SPEED, vmax + 3.0)
        self.long_acc = a

        # ---------------- real vehicle dynamics & tire physics ----------------
        grip = 1.0
        surf_drag = 1.0
        is_grass = False
        if city is not None:
            code, _ = city.surface_at(self.x, self.y, self.lvl)
            if code in (1, 2):    # Paved asphalt road / highway / landmark driveway
                grip = 1.0
                surf_drag = 1.0
            elif code == 3:       # Paved sidewalk
                grip = 0.85
                surf_drag = 1.15
            else:                 # Grass / unpaved terrain
                grip = 0.72
                surf_drag = 1.35
                is_grass = True
        self.tire_grip = grip

        # Off-road grass physics: allows vehicle to always drive and recover back to road without getting stuck
        if is_grass:
            v_cap = 6.94  # ~25 km/h off-road limit on grass terrain
            if abs(self.v) > v_cap:
                self.v -= math.copysign(3.5 * dt, self.v)
            elif self.throttle < 0.05 and (not in_rev or self.brake < 0.05) and abs(self.v) > 0.1:
                # Natural coasting rolling resistance when no pedal is pressed
                self.v -= math.copysign(1.5 * dt, self.v)
        elif surf_drag > 1.0 and abs(self.v) > 0.1:
            if self.throttle < 0.05 and (not in_rev or self.brake < 0.05):
                self.v -= math.copysign((surf_drag - 1.0) * 2.5 * dt, self.v)

        # Dynamic tire friction limit and lateral grip
        v_ref = max(av, 1.5)
        a_lat_max = grip * 8.8  # saturated lateral acceleration limit (m/s^2)
        raw_yaw = (self.v / self.wheelbase) * math.tan(self.steer_angle)
        raw_lat = raw_yaw * self.v

        if abs(raw_lat) > a_lat_max:
            yaw = math.copysign(a_lat_max / v_ref, raw_lat)
            lat_acc = math.copysign(a_lat_max, raw_lat)
            # Tire scrub: cornering at the grip limit bleeds speed naturally
            self.v -= math.copysign(0.045 * abs(raw_lat - a_lat_max) * dt, self.v)
        else:
            yaw = raw_yaw
            lat_acc = raw_lat

        self.yaw_rate = yaw
        self.lat_acc = lat_acc
        self.heading = normalize_angle(self.heading + yaw * dt)
        self.x += math.cos(self.heading) * self.v * dt
        self.y += math.sin(self.heading) * self.v * dt

        # Suspension compliance: chassis body roll in turns, nose dive under braking, squat under acceleration
        tgt_roll = -clamp(lat_acc / 9.81, -1.2, 1.2) * 0.075
        tgt_pitch = clamp(self.long_acc / 9.81, -1.0, 0.8) * 0.045
        self.roll += (tgt_roll - self.roll) * min(1.0, 14.0 * dt)
        self.pitch += (tgt_pitch - self.pitch) * min(1.0, 14.0 * dt)

        self.norm_steer = clamp(self.steer_angle / self.max_steer, -1.0, 1.0)
        self.norm_throttle = clamp(self.throttle - self.brake, -1.0, 1.0)

    def pose_ahead(self, dist):
        return (self.x + math.cos(self.heading) * dist, self.y + math.sin(self.heading) * dist, self.heading)

    def adjust_cruise(self, delta):
        if not self.cruise_on:
            self.cruise_on = True
            self.cruise_i = 0.12
            self.cruise_v = max(self.v, 5.0)
        self.cruise_v = clamp(self.cruise_v + delta, 4.0, Config.MAX_SPEED)

    def toggle_cruise(self):
        if self.cruise_on:
            self.cruise_on = False
        elif self.v > 4.0:
            self.cruise_on = True
            self.cruise_i = 0.12
            self.cruise_v = min(self.v, Config.MAX_SPEED)
        return self.cruise_on


# =============================================================================
# DYNAMIC AGENT GRID  (cars, player, pedestrians)
# =============================================================================
class AgentGrid:
    def __init__(self, cell=30.0):
        self.cell = cell
        self.cells = defaultdict(list)

    def clear(self):
        self.cells.clear()

    def add(self, a):
        k = (int(a.x // self.cell), int(a.y // self.cell))
        self.cells[k].append(a)

    def near(self, x, y, r):
        c = self.cell
        out = []
        for gx in range(int((x - r) // c), int((x + r) // c) + 1):
            for gy in range(int((y - r) // c), int((y + r) // c) + 1):
                lst = self.cells.get((gx, gy))
                if lst:
                    out.extend(lst)
        return out


# =============================================================================
# AI TRAFFIC
# =============================================================================
class AICar:
    kind = 'car'
    z = 0.0
    S0, T_HW, A_MAX, B_COMF = 2.4, 1.25, 2.2, 3.0

    def __init__(self, uid, path, s, speed, colors, body, rng):
        self.id = uid
        self.rng = rng
        self.path = path
        self.s = s
        self.v = speed
        self.color, self.color2 = colors
        self.body = body
        spec = VEHICLES[body]
        self.vkind = spec['kind']
        self.length = spec['length']
        self.width = spec['width']
        self.pace = rng.uniform(*spec['pace'])
        self.A_MAX, self.B_COMF, self.S0 = spec['a_max'], spec['b'], spec['s0']
        self.T_HW = 1.0 if self.vkind in ('bike', 'scooter') else (1.6 if self.vkind in ('bus', 'truck') else 1.25)
        self.lat = rng.uniform(-0.45, 0.45) if self.vkind in ('bike', 'scooter') else 0.0
        self.rider = rng.choice(RIDER_COLORS)
        self.braking = False
        self.nxt = None
        self.stuck = 0.0
        self.still = 0.0
        self.ghost = 0.0
        self.x = self.y = self.heading = 0.0
        self.prio = 2
        self.stopped_for_signal = False
        self.route_map = {}
        self.need_plan = True
        self.entering = False
        self._choose_next()
        self._update_pose()

    # ---- route -----------------------------------------------------------
    def _choose_next(self):
        p = self.path
        if p.kind == 'conn':
            self.nxt = p.out[0]
        else:
            opts = p.out
            if len(opts) == 1:
                self.nxt = opts[0]
            else:
                pick = self.route_map.get(p.pid)
                if pick is not None and pick in opts:
                    self.nxt = pick
                else:
                    self.need_plan = True
                    cand = opts
                    if self.vkind in ('bus', 'truck'):
                        lim = 9.0 if self.vkind == 'bus' else 7.0
                        c2 = [c for c in opts if c.move == 'S' or c.rmin >= lim]
                        if c2:
                            cand = c2
                    hw = p.edge.rt.rank >= 2
                    w = [((0.8 if hw else 0.55) if c.move == 'S' else (0.1 if hw else 0.225)) / max(1, getattr(c, 'dups', 1)) for c in cand]
                    self.nxt = self.rng.choices(cand, weights=w)[0]
        self.prio = 1 if (self.path.kind == 'conn' and self.path.move == 'L') or \
            (self.path.kind == 'lane' and self.nxt is not None and self.nxt.move == 'L'
             and self.path.length - self.s < 30.0) else 2

    def _update_pose(self):
        x, y, h = self.path.pose(self.s)
        if self.lat:
            x += -math.sin(h) * self.lat
            y += math.cos(h) * self.lat
        self.x, self.y, self.heading = x, y, h
        self.z = self.path.zat(self.s) if self.path.elev is not None else 0.0

    def corners(self):
        return obb_corners(self.x, self.y, self.heading, self.length, self.width)

    def pose_ahead(self, dist):
        """Pose after travelling `dist` further along the planned route."""
        s = self.s + dist
        p = self.path
        if s <= p.length:
            return p.pose(s)
        s -= p.length
        n = self.nxt
        if n is not None:
            if s <= n.length:
                return n.pose(s)
            s -= n.length
            if n.out:
                o = n.out[0]
                return o.pose(min(s, o.length))
            return n.pose(n.length)
        return p.pose(p.length)

    def route_samples(self, dist, step=2.0):
        """Points ahead of the front bumper along the planned route -> [(x, y, d)]"""
        out = []
        path, s = self.path, self.s + self.length * 0.5
        nxt = self.nxt
        d = 0.0
        stage = 0
        cur = path
        while d <= dist:
            if s > cur.length:
                s -= cur.length
                if stage == 0:
                    cur, stage = nxt, 1
                elif stage == 1 and cur.out:
                    cur, stage = cur.out[0], 2
                else:
                    break
                if cur is None:
                    break
                continue
            px, py, _ = cur.pose(s)
            out.append((px, py, d))
            s += step
            d += step
        return out


AI_BODIES = ['sedan', 'suv', 'truck', 'bus', 'auto', 'bike', 'scooter']


class _PlayerOcc:
    """Stand-in that lets AI traffic treat the player as a junction occupant."""
    kind = 'player_occ'
    entering = False
    nxt = None
    body = 'sedan'
    stopped_for_signal = False
    id = -1

    def __init__(self):
        self.path = None
        self.length = 4.5
        self.v = 0.0


class TrafficManager:
    def __init__(self, city, rng_seed=None):
        self.city = city
        self.rng = random.Random(rng_seed if rng_seed is not None else 777)
        self.cars = []
        self.uid = 0
        self.grid = AgentGrid(30.0)
        self.t = 0.0
        self.step_i = 0
        self._bposes = {}
        self._pocc = _PlayerOcc()
        self._seed_t = 0.0
        self._pool = city.main_lanes
        self._peds = []
        w = [l.length for l in self._pool]
        self._cum = []
        acc = 0.0
        for x in w:
            acc += x
            self._cum.append(acc)
        self._tot = acc

    def reset(self, player_xy=None, count=None, seed_lane=None, seed_s=0.0):
        self.cars = []
        n = Config.TRAFFIC_COUNT if count is None else count
        tries = 0
        while len(self.cars) < n and tries < n * 40:
            tries += 1
            self._spawn_stream(player_xy, None, True)
        if seed_lane is not None:
            self._seed_ahead(seed_lane, seed_s)

    def _pick_lane(self):
        r = self.rng.uniform(0, self._tot)
        i = bisect.bisect_left(self._cum, r)
        return self._pool[min(i, len(self._pool) - 1)]

    def _place(self, lane, s, speed_frac=0.5):
        x, y, h = lane.pose(s)
        for c in self.cars:
            if abs(c.x - x) < 17 and abs(c.y - y) < 17 and abs(c.z - lane.zat(s)) < 3.0:
                return None
        for p in self._peds:
            if abs(p.x - x) < 9.0 and abs(p.y - y) < 9.0:
                return None
        for cw in self.city.crosswalks:
            if abs(cw.cx - x) < 7.0 and abs(cw.cy - y) < 7.0:
                return None
        names = list(AI_BODIES)
        weights = [VEHICLES[k]['ai_w'] for k in names]
        key = lane.edge.rt.key
        for _ in range(4):
            body = self.rng.choices(names, weights=weights)[0]
            if key == 'L' and body in ('bus', 'truck'):
                continue
            break
        else:
            body = 'scooter'
        if key == 'H' and body in ('scooter', 'auto') and self.rng.random() < 0.7:
            body = 'sedan'
        colors = self._colors(body)
        self.uid += 1
        car = AICar(self.uid, lane, s, lane.limit * speed_frac, colors, body, self.rng)
        self.cars.append(car)
        return car

    def _spawn_stream(self, pxy, heading, initial=False):
        """Create one vehicle: anywhere around the player at start, on the outer ring (out of sight) afterwards."""
        lane = self._pick_lane()
        s = self.rng.uniform(6.0, max(7.0, lane.length - 12.0))
        x, y, h = lane.pose(s)
        if pxy is not None:
            dx, dy = x - pxy[0], y - pxy[1]
            d = math.hypot(dx, dy)
            R = Config.TRAFFIC_RADIUS
            if initial:
                if d < 28.0 or d > R:
                    return False
            else:
                if d > R * 0.97 or d < R * 0.55:
                    return False
                if heading is not None and d < R * 0.85:
                    if abs(normalize_angle(math.atan2(dy, dx) - heading)) < 0.95:
                        return False        # would pop into view ahead
        return self._place(lane, s, 0.55) is not None

    def _seed_ahead(self, lane, s, d_from=34.0, d_to=260.0):
        """Make sure there is traffic in the player's own lane ahead AND in the oncoming lane."""
        rng = self.rng
        d = rng.uniform(d_from, d_from + 18.0)
        while d < d_to:
            p, ps, rem = lane, s, d
            hops = 0
            while ps + rem > p.length and hops < 6:
                rem -= (p.length - ps)
                conn = None
                for c in p.out:
                    if c.move == 'S':
                        conn = c
                        break
                if conn is None and p.out:
                    conn = p.out[0]
                if conn is None:
                    break
                rem -= conn.length
                p = conn.out_lane
                ps = 0.0
                hops += 1
            if rem < 0.0 or ps + rem > p.length - 4.0:
                d += rng.uniform(26.0, 52.0)
                continue
            pos = ps + rem
            lanes_same = p.edge.lanes.get(p.tdir, [p])
            if rng.random() < 0.85:
                ln = rng.choice(lanes_same) if len(lanes_same) > 1 and rng.random() < 0.5 else p
                self._place(ln, pos if ln is p else clamp(pos, 8.0, ln.length - 8.0), 0.6)
            # oncoming lane(s): same position along the road
            opp = p.edge.lanes.get(OPP[p.tdir])
            if opp:
                u = p.u0 + p.du * pos
                for ol in opp:
                    so = (u - ol.u0) / ol.du + rng.uniform(-14.0, 14.0)
                    if 8.0 < so < ol.length - 8.0 and rng.random() < 0.8:
                        self._place(ol, so, 0.6)
            d += rng.uniform(26.0, 52.0)

    def _colors(self, body):
        r = self.rng
        if body == 'bus':
            return r.choice(BUS_COLORS)
        if body == 'auto':
            return r.choice(AUTO_COLORS)
        if body == 'truck':
            return (r.choice([(235, 235, 232), (60, 90, 150), (190, 60, 50), (70, 120, 80)]), (50, 50, 56))
        if body in ('bike', 'scooter'):
            return (r.choice(CAR_COLORS), (40, 40, 46))
        return (r.choice(CAR_COLORS), (30, 30, 36))

    def _plan(self, car):
        """Give an AI vehicle a random far-away destination and the route to it."""
        city = self.city
        start = car.path if car.path.kind == 'lane' else car.path.out_lane
        best, bd = None, -1.0
        for _ in range(4):
            g = city.main_lanes[self.rng.randrange(len(city.main_lanes))]
            if g.length < 50.0 or (g.edge.rt.key == 'L' and car.vkind in ('bus', 'truck')):
                continue
            d = math.hypot(g.pts[0][0] - start.pts[0][0], g.pts[0][1] - start.pts[0][1])
            if d > bd:
                best, bd = g, d
        car.need_plan = False
        car.route_map = {}
        if best is None:
            return
        lim = 9.0 if car.vkind == 'bus' else (7.0 if car.vkind == 'truck' else 0.0)
        route = city.plan_route(start, best, lim)
        if not route:
            return
        for k in range(len(route) - 1):
            if route[k].kind == 'lane':
                car.route_map[route[k].pid] = route[k + 1]

    # ---------------------------------------------------------------- brain
    def _idm(self, car, v0, leaders):
        v = car.v
        a_free = car.A_MAX * (1.0 - (v / max(v0, 0.5)) ** 4)
        a = a_free
        sq = 2.0 * math.sqrt(car.A_MAX * car.B_COMF)
        for gap, vl in leaders:
            gap = max(gap, 0.05)
            s_star = car.S0 + max(0.0, v * car.T_HW + v * (v - vl) / sq)
            ai = car.A_MAX * (1.0 - (v / max(v0, 0.5)) ** 4 - (s_star / gap) ** 2)
            if ai < a:
                a = ai
        return max(a, -8.0)

    def _perceive(self, car, player, peds):
        """Return list of (gap, leader_speed) constraints for this car."""
        leaders = []
        v = car.v
        D = max(10.0, v * 2.4 + v * v / 6.0 + 6.0)
        if car.prio == 1:
            D = max(D, 26.0)
        samples = car.route_samples(D)
        if not samples:
            return leaders
        cx, cy = samples[0][0], samples[0][1]
        near = self.grid.near(car.x, car.y, D + 8.0)
        hwA = max(car.width * 0.5, car.length * 0.2)
        # candidates (pre-filter by same agent id)
        for B in near:
            if B is car:
                continue
            if abs(B.z - car.z) > 2.6:
                bp = getattr(B, 'path', None)
                nx = car.nxt
                if bp is None or not (bp is car.path or bp is nx or
                                      (nx is not None and nx.kind == 'conn' and bp is nx.out_lane)):
                    continue
            if B.kind == 'ped':
                if B.state == 'wait':
                    # Pedestrians standing still on sidewalk waiting for green light must NOT block vehicles!
                    continue
                rB = 0.35 + hwA + 0.55
                hit_d = None
                for (px, py, d) in samples:
                    if (px - B.x) ** 2 + (py - B.y) ** 2 < rB * rB:
                        hit_d = d
                        break
                if hit_d is None and B.state == 'cross':
                    v_eff = max(v, 2.5)
                    for (px, py, d) in samples:
                        tau = d / v_eff
                        for f in (0.6, 1.0, 1.4, 1.8):
                            bx, by = B.pos_after(max(0.0, tau * f + (0.4 if f > 1.0 else -0.2)))
                            if (px - bx) ** 2 + (py - by) ** 2 < rB * rB:
                                hit_d = d
                                break
                        if hit_d is not None:
                            break
                # pedestrian right beside / in front of my body
                if hit_d is None or hit_d > 0.5:
                    chh, shh = math.cos(car.heading), math.sin(car.heading)
                    ddx, ddy = B.x - car.x, B.y - car.y
                    lx = ddx * chh + ddy * shh
                    ly = -ddx * shh + ddy * chh
                    if abs(lx) < car.length * 0.5 + 0.8 and abs(ly) < hwA + 0.9:
                        hit_d = 0.55
                if hit_d is not None:
                    leaders.append((max(hit_d - 0.5, 0.05), 0.0))
                continue
            # vehicles
            same_dir = math.cos(B.heading - car.heading) > 0.55
            ch, sh = math.cos(B.heading), math.sin(B.heading)
            hl = B.length * 0.5 + (0.6 if same_dir else 0.9)
            hw = B.width * 0.5 + hwA + (0.25 if same_dir else 0.45)
            hit_d = None
            if same_dir or car.ghost <= 0.0:
                for (px, py, d) in samples:
                    dx, dy = px - B.x, py - B.y
                    lx = dx * ch + dy * sh
                    if -hl < lx < hl:
                        ly = -dx * sh + dy * ch
                        if -hw < ly < hw:
                            hit_d = d
                            break
            if hit_d is not None:
                leaders.append((max(hit_d, 0.1), B.v if same_dir else (0.0 if abs(B.v) < 1.0 else B.v * 0.5)))
                continue
            if same_dir or car.path.kind == 'conn' or car.ghost > 0.0:
                continue
            # ---- predicted conflicts (only before entering the junction) ----
            if B.kind == 'player':
                vp = abs(B.v)
                if vp < 0.8:
                    continue
            else:
                has_way = car.prio > B.prio or (car.prio == B.prio and car.id < B.id)
                if car.prio != 1 or has_way:
                    continue
                vp = abs(B.v)
                if vp < 0.6:
                    if not B.stopped_for_signal and (B.path.kind == 'conn' or B.path.length - B.s < 8.0):
                        vp = 3.5
                    else:
                        continue
            v_eff = max(v, 3.8)
            poses = self._bposes.get(B.id)
            if poses is None:
                poses = [B.pose_ahead(vp * 0.4 * k) for k in range(24)]
                self._bposes[B.id] = poses
            rr = (B.length * 0.5 + hwA + 1.2) ** 2
            for (px, py, d) in samples:
                if d < 0.5:
                    continue
                tau = d / v_eff
                k0 = max(0, int((0.7 * tau - 0.3) / 0.4))
                k1 = min(23, int((1.9 * tau + 0.8) / 0.4) + 1)
                hit = False
                for k in range(k0, k1 + 1):
                    bx, by, bh = poses[k]
                    ddx, ddy = px - bx, py - by
                    if ddx * ddx + ddy * ddy > rr:
                        continue
                    c2, s2 = math.cos(bh), math.sin(bh)
                    lx = ddx * c2 + ddy * s2
                    ly = -ddx * s2 + ddy * c2
                    if abs(lx) < B.length * 0.5 + 0.9 and abs(ly) < B.width * 0.5 + hwA + 0.45:
                        hit = True
                        break
                if hit:
                    leaders.append((max(d - 0.3, 0.1), 0.0))
                    break
        return leaders

    def update(self, dt, t_now, player, peds):
        self.t = t_now
        self.step_i += 1
        self._bposes = {}
        self._peds = peds
        grid = self.grid
        grid.clear()
        for n in self.city.nodes.values():
            if n.occ:
                n.occ = []
        for c in self.cars:
            grid.add(c)
            if c.path.kind == 'conn' and c.path.node.kind == 'junction':
                c.path.node.occ.append(c)
            elif c.entering:
                c.nxt.node.occ.append(c)
        pres = self.city.nearest_path(player.x, player.y, player.heading, player.z, 5.0)
        if pres is not None and pres[0].kind == 'conn' and pres[0].node.kind == 'junction':
            po = self._pocc
            po.path, po.length, po.v = pres[0], player.length, player.v
            pres[0].node.occ.append(po)
        grid.add(player)
        for p in peds:
            grid.add(p)

        rng = self.rng
        keep = []
        budget = 3
        by_lane = defaultdict(list)
        for c in self.cars:
            if c.path.kind == 'lane':
                by_lane[c.path.pid].append(c)
        self._by_lane = by_lane
        for car in self.cars:
            if car.need_plan and budget > 0:
                budget -= 1
                self._plan(car)
            path = car.path
            v0 = path.limit * car.pace
            # curve pre-slowing
            if path.kind == 'lane' and car.nxt is not None and car.nxt.cap < v0:
                rem = max(0.0, path.length - car.s - car.length * 0.5)
                v0 = min(v0, math.sqrt(car.nxt.cap ** 2 + 2.0 * 2.2 * rem))
            elif path.kind == 'conn':
                v0 = min(v0, path.cap)

            # perception (every 2nd step, staggered)
            if (self.step_i + car.id) % 2 == 0 or not hasattr(car, '_ld'):
                leaders = self._perceive(car, player, peds)
                # traffic signal
                car.stopped_for_signal = False
                if path.kind == 'lane' and path.signal is not None:
                    d = path.stop_s - (car.s + car.length * 0.5)
                    if d > -1.2:
                        st = path.signal.state(path.axis, t_now)
                        stop = False
                        if st == 'R':
                            stop = True
                        elif st == 'Y':
                            stop = d > (car.v * car.v) / (2.0 * 3.2) + 1.0
                        if stop:
                            leaders.append((max(d, 0.05), 0.0))
                            car.stopped_for_signal = True
                        elif d < 14.0 and car.nxt is not None:
                            # don't block the box: out-lane start must have room
                            ol = car.nxt.out[0] if car.nxt.out else None
                            if ol is not None:
                                ox, oy, _ = ol.pose(6.0)
                                for B in grid.near(ox, oy, 9.0):
                                    if B is car or B.kind == 'ped':
                                        continue
                                    if abs(B.v) < 1.5 and (B.x - ox) ** 2 + (B.y - oy) ** 2 < 100.0 and \
                                            B.kind == 'car' and B.path is ol:
                                        leaders.append((max(d, 0.05), 0.0))
                                        break
                car._ld = leaders
            leaders = car._ld
            # ---- junction entry: wait while a conflicting movement is inside the box ----
            if path.kind == 'lane' and car.nxt is not None and path.node.kind == 'junction':
                d_end = path.length - (car.s + car.length * 0.5)
                if d_end < max(14.0, car.v * car.v / 9.0 + 6.0):
                    blocked = False
                    if car.ghost <= 0.0:
                        for o in path.node.occ:
                            cf = car.nxt.conflicts_wide if (car.length > 6.0 or o.length > 6.0) else car.nxt.conflicts
                            if o is not car and self._eff(o).pid in cf:
                                blocked = True
                                break
                        if not blocked and self._exit_blocked(car):
                            blocked = True
                    if blocked:
                        leaders = leaders + [(max(d_end - 0.4, 0.05), 0.0)]
            a = self._idm(car, v0, leaders)
            car.braking = a < -0.6 or (car.v < 0.3 and bool(leaders))
            car.v = max(0.0, car.v + a * dt)
            car.s += car.v * dt

            # stuck / deadlock breaking
            if car.v < 0.25 and not car.stopped_for_signal and \
                    not any(g[1] == 0.0 and g[0] < 12 for g in leaders if False):
                if car.path.kind == 'conn':
                    car.stuck += dt * 3.5  # Intersection blockages clear rapidly
                else:
                    car.stuck += dt
            else:
                car.stuck = max(0.0, car.stuck - dt * 2.0)
            if car.path.kind == 'conn' and car.stuck > 2.5 and car.ghost <= 0.0:
                car.ghost = 4.0
                car.v = max(car.v, 2.2)  # Creep forward to clear the intersection box
            elif car.stuck > 14.0 and car.ghost <= 0.0:
                nd = car.path.node if car.path.kind == 'lane' else car.path.node
                owner = getattr(nd, 'ghost_owner', None)
                if owner is None or owner.ghost <= 0.0 or owner is car:
                    nd.ghost_owner = car          # only ONE vehicle per junction may break a jam at a time
                    car.ghost = 7.0
                    car.stuck = 8.0
            if car.ghost > 0.0:
                car.ghost -= dt

            # advance along route (a vehicle claims the junction when its NOSE reaches the line)
            if path.kind == 'lane' and car.nxt is not None and car.nxt.kind == 'conn' \
                    and car.nxt.node.kind == 'junction' and not car.entering and car.ghost <= 0.0:
                if car.s + car.length * 0.5 >= path.length - 0.15:
                    hold = False
                    for o in car.nxt.node.occ:
                        cf = car.nxt.conflicts_wide if (car.length > 6.0 or o.length > 6.0) else car.nxt.conflicts
                        if o is not car and self._eff(o).pid in cf:
                            hold = True
                            break
                    if not hold and self._exit_blocked(car):
                        hold = True
                    if hold:
                        car.s = path.length - car.length * 0.5 - 0.2
                        car.v = 0.0
                    else:
                        car.entering = True
                        car.nxt.node.occ.append(car)
            if car.s >= path.length:
                car.s -= path.length
                car.path = car.nxt
                car.entering = False
                if car.path is not None and car.path.kind == 'conn' and car.path.node.kind == 'junction':
                    if car not in car.path.node.occ:
                        car.path.node.occ.append(car)
                car._choose_next()
                if car.path is None:
                    continue
            car._update_pose()
            if car.v < 0.3:
                car.still += dt
            else:
                car.still = 0.0
            if car.still > 85.0 and not car.stopped_for_signal:
                continue            # frozen far longer than any red light: remove (a new one is spawned elsewhere)
            keep.append(car)
        px, py = player.x, player.y
        lim2 = (Config.TRAFFIC_RADIUS * 1.12) ** 2
        self.cars = [c for c in keep if (c.x - px) ** 2 + (c.y - py) ** 2 < lim2]
        self._separate()
        # keep feeding traffic onto the road the player is driving along (same lane + oncoming), far ahead
        self._seed_t -= dt
        if self._seed_t <= 0.0:
            self._seed_t = 0.8
            res = self.city.nearest_path(player.x, player.y, player.heading, player.z, 9.0)
            if res is not None:
                pth, ps = res[0], res[1]
                if pth.kind == 'conn':
                    pth, ps = pth.out_lane, 0.0
                if pth.main:
                    self._seed_ahead(pth, ps, 235.0, 335.0)
        # keep the population around the player topped up (spawned out of sight)
        tries = 0
        while len(self.cars) < Config.TRAFFIC_COUNT and tries < 40:
            tries += 1
            self._spawn_stream((px, py), player.heading, False)

    @staticmethod
    def _eff(o):
        return o.path if o.path.kind == 'conn' else o.nxt

    def _exit_blocked(self, car):
        """'Don't block the box': is there no room for this vehicle on the road beyond the junction?"""
        nx = car.nxt
        if nx is None or nx.kind != 'conn' or nx.node.kind != 'junction':
            return False
        ol = nx.out_lane
        head, hs = None, 1e9
        for c in self._by_lane.get(ol.pid, ()):
            if c is car:
                continue
            e = c.s - c.length * 0.5
            if e < hs:
                hs, head = e, c
        if head is None:
            return False
        pending = 0.0
        for o in nx.node.occ:
            if o is not car and self._eff(o).out_lane is ol:
                pending += o.length + 2.5
        return hs < car.length + 2.5 + pending and head.v < 4.0

    def _separate(self):
        """Collision response: vehicles that touch brake hard (the one behind / lower priority)."""
        cell = 12.0
        g = defaultdict(list)
        for c in self.cars:
            g[(int(c.x // cell), int(c.y // cell))].append(c)
        for c in self.cars:
            gx, gy = int(c.x // cell), int(c.y // cell)
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for o in g.get((gx + ox, gy + oy), ()):
                        if o.id <= c.id or abs(o.z - c.z) > 2.5:
                            continue
                        r = (c.length + o.length) * 0.5 + 0.6
                        dx, dy = o.x - c.x, o.y - c.y
                        if dx * dx + dy * dy > r * r:
                            continue
                        if not sat_polygon_collision(c.corners(), o.corners()):
                            continue
                        ahead_c = dx * math.cos(c.heading) + dy * math.sin(c.heading) < 0   # c is ahead of o
                        if abs(math.cos(c.heading - o.heading)) > 0.5:
                            rear, front = (o, c) if ahead_c else (c, o)
                        else:
                            rear, front = (c, o) if c.id > o.id else (o, c)
                        rear.v = min(rear.v * 0.35, max(0.0, front.v * 0.7))
                        rear.braking = True

    def clear_around(self, x, y, r, ahead_ok=False):
        self.cars = [c for c in self.cars if (c.x - x) ** 2 + (c.y - y) ** 2 > r * r]


# =============================================================================
# PEDESTRIANS
# =============================================================================
class Pedestrian:
    kind = 'ped'
    z = 0.0
    SHIRTS = [(225, 48, 48), (42, 115, 220), (38, 160, 75), (240, 165, 30),
              (160, 65, 195), (230, 105, 40), (240, 240, 245), (30, 30, 36)]
    v = 0.0
    heading = 0.0
    prio = 9
    id = 0

    def __init__(self, cw, side, rng):
        self.cw = cw
        self.rng = rng
        self.side = side                    # -1 / +1 : which kerb we start on
        self.lat = side * cw.wait_off       # lateral position across the road
        self.along = rng.uniform(-1.0, 1.0)
        self.speed = rng.uniform(*Config.PED_SPEED)
        self.state = 'wait'
        self.wait_t = 0.0
        self.shirt = rng.choice(self.SHIRTS)
        self.pants = rng.choice([(42, 48, 62), (70, 70, 80), (30, 50, 90), (90, 80, 60)])
        self.phase = rng.random() * 6.28
        self.height = rng.uniform(1.55, 1.85)
        self.done = False
        self.leave_dist = 0.0
        self._upd()

    def _upd(self):
        cw = self.cw
        self.x = cw.cx + cw.vx * self.lat + cw.ux * self.along
        self.y = cw.cy + cw.vy * self.lat + cw.uy * self.along
        if self.state == 'cross':
            self.heading = math.atan2(-self.side * cw.vy, -self.side * cw.vx)
        elif self.state == 'leave':
            self.heading = math.atan2(-self.side * cw.vy, -self.side * cw.vx)
        else:
            self.heading = math.atan2(cw.vy * -self.side, cw.vx * -self.side)

    blocked = False

    @property
    def moving(self):
        return self.state in ('cross', 'leave') and not self.blocked

    def pos_after(self, t):
        """Predicted (x, y) after t seconds, walking on towards the far kerb."""
        cw = self.cw
        if self.state == 'wait':
            return self.x, self.y
        sp = self.speed if self.state == 'cross' else self.speed * 0.5
        lat = self.lat - self.side * sp * t
        far = -self.side * (cw.wait_off + (3.0 if self.state == 'leave' else 0.0))
        if (lat - far) * self.side < 0:
            lat = far
        return cw.cx + cw.vx * lat + cw.ux * self.along, cw.cy + cw.vy * lat + cw.uy * self.along

    def update(self, dt, can_go):
        cw = self.cw
        if self.blocked and self.state != 'wait':
            return
        self.phase += dt * 6.5
        if self.state == 'wait':
            self.wait_t += dt
            if can_go:
                self.state = 'cross'
        elif self.state == 'cross':
            self.lat -= self.side * self.speed * dt
            if self.lat * self.side < -cw.wait_off + 0.2:
                self.state = 'leave'
        elif self.state == 'leave':
            self.lat -= self.side * self.speed * 0.5 * dt
            self.along += 0.4 * self.speed * dt * (1 if self.rng.random() < 0.5 else 1)
            self.leave_dist += self.speed * 0.5 * dt
            if self.leave_dist > 3.0:
                self.done = True
        self._upd()


class PedestrianManager:
    def __init__(self, city, seed=None):
        self.city = city
        self.rng = random.Random((seed if seed is not None else 5) * 31 + 7)
        self.peds = []
        self.timer = 0.0

    def reset(self):
        for cw in self.city.crosswalks:
            cw.ped = None
            cw.timer = self.rng.uniform(0.5, 14.0)
        self.peds = []

    def _signal_ok(self, cw, t):
        sig = cw.signal
        if sig is None:
            return True
        # STRICT RULE: Vehicle light must be RED
        if sig.state(cw.road_axis, t) != 'R':
            return False
        # Calculate how much continuous RED time remains for vehicles on this road
        red_remaining = 0.0
        tt = t
        step = 0.25
        max_check = int(sig.cycle / step) + 4
        for _ in range(max_check):
            tt += step
            if sig.state(cw.road_axis, tt) != 'R':
                red_remaining = tt - t
                break
        else:
            red_remaining = sig.cycle
        # Pedestrian needs to cross road width + safety buffer
        cross_time = (2.0 * cw.road_half) / 1.2 + 2.5
        return red_remaining >= cross_time

    def update(self, dt, t_now, player, traffic):
        rng = self.rng
        city = self.city
        R = Config.PED_ACTIVE_RADIUS
        # spawn
        if len(self.peds) < Config.PED_MAX:
            for cw in city.crosswalks:
                if cw.ped is None:
                    cw.timer -= dt
                    if cw.timer <= 0.0 and (cw.cx - player.x) ** 2 + (cw.cy - player.y) ** 2 < R * R:
                        p = Pedestrian(cw, rng.choice([-1, 1]), rng)
                        cw.ped = p
                        self.peds.append(p)
                        cw.timer = rng.uniform(5.0, 20.0)
                        if len(self.peds) >= Config.PED_MAX:
                            break
        alive = []
        cars = traffic.cars
        for p in self.peds:
            cw = p.cw
            can_go = False
            if p.state == 'wait':
                if cw.signal is not None:
                    can_go = self._signal_ok(cw, t_now) and self._midblock_clear(cw, player, cars)
                else:
                    can_go = self._midblock_clear(cw, player, cars) and p.wait_t > 1.2
            elif p.state == 'cross':
                # If vehicle signal turned GREEN or YELLOW while crossing, HURRY to clear!
                if cw.signal is not None and cw.signal.state(cw.road_axis, t_now) != 'R':
                    p.speed = max(p.speed, 2.6)
            was_blocked = p.blocked
            p.blocked = False
            if p.state != 'wait':
                dxp, dyp = -p.side * cw.vx, -p.side * cw.vy
                qx, qy = p.x + dxp * 1.0, p.y + dyp * 1.0
                for c in cars + [player]:
                    if abs(c.x - qx) > 5.0 or abs(c.y - qy) > 5.0:
                        continue
                    ch, sh = math.cos(c.heading), math.sin(c.heading)
                    ex, ey = qx - c.x, qy - c.y
                    if abs(ex * ch + ey * sh) < c.length * 0.5 + 0.45 and abs(-ex * sh + ey * ch) < c.width * 0.5 + 0.45:
                        p.blocked = True
                        break
            if p.blocked:
                p.blocked_t = getattr(p, 'blocked_t', 0.0) + dt
                if p.blocked_t > 2.5:
                    p.blocked = False           # gave the car enough time; walk on
            else:
                p.blocked_t = 0.0
            p.update(dt, can_go)
            far = (cw.cx - player.x) ** 2 + (cw.cy - player.y) ** 2 > (R * 1.4) ** 2
            if p.done or (far and p.state == 'wait'):
                cw.ped = None
                cw.timer = max(cw.timer, rng.uniform(4.0, 16.0))
                continue
            alive.append(p)
        self.peds = alive

    @staticmethod
    def _midblock_clear(cw, player, cars):
        """True when no vehicle is in, or about to reach, the crosswalk."""
        signalised = cw.signal is not None
        for c in cars + [player]:
            dx, dy = c.x - cw.cx, c.y - cw.cy
            d = math.hypot(dx, dy)
            if d > 65.0:
                continue
            speed = abs(c.v)
            along = dx * cw.ux + dy * cw.uy
            across = dx * cw.vx + dy * cw.vy
            # physically on / touching the zebra
            if abs(along) < 1.5 + c.length * 0.5 + 0.5 and abs(across) < cw.road_half + 0.6 and \
                    (speed > 0.3 or abs(along) < 1.5 + 1.0):
                return False
            if signalised and getattr(c, 'stopped_for_signal', False):
                continue
            if signalised and speed < 0.5 and d > 3.5:
                continue
            hx, hy = math.cos(c.heading), math.sin(c.heading)
            towards = (cw.cx - c.x) * hx + (cw.cy - c.y) * hy
            if signalised:
                if speed > 1.2 and (towards > -2.0 or d < 18.0):
                    return False
            else:
                if (towards > -2.0 or d < 18.0) and d < 34.0 + 3.0 * speed:
                    return False
        return True


# =============================================================================
# AIRPLANE SIMULATION (State Machine: Approach -> Rollout -> Turn -> Wait -> Takeoff -> Gone)
# =============================================================================
class AirportAirplane:
    """
    Simulates airliner flight, landing flare, rollout, turnaround, engine wait,
    accelerated takeoff, climb, gear retraction, and circuit looping.
    Matches the Ursina airport state machine and physical flight specifications.
    """
    def __init__(self, airport_dest):
        self.airport_dest = airport_dest
        self.curb_pt = airport_dest['curb_pt']
        self.finish_pt = airport_dest['finish_pt']
        self.b_center = airport_dest['b_center']
        self.f_vec = airport_dest['f_vec']
        self.out_vec = airport_dest['out_vec']

        # Runway center & specs (parallel to f_vec, offset behind terminal)
        self.rw_center_x = self.b_center[0] + self.out_vec[0] * 55.0
        self.rw_center_y = self.b_center[1] + self.out_vec[1] * 55.0
        self.rw_len = 520.0
        self.rw_width = 30.0

        self.TOUCHDOWN_S = 100.0
        self.START_S = 380.0
        self.START_ALT = 80.0
        self.GROUND_ALT = 4.2
        self.reset()

    def reset(self):
        self.state = 'approach'
        self.timer = 0.0
        self.speed = 48.0
        self.dir = -1.0
        self.s = self.START_S
        self.lat = 0.0
        self.alt = self.START_ALT
        self.pitch = -6.0
        self.yaw = 180.0
        self.gear_down = True
        self.visible = True
        self._sync_world()

    def _sync_world(self):
        s_offset = self.s - (self.rw_len * 0.5)
        fx, fy = self.f_vec
        out_x, out_y = self.out_vec
        self.x = self.rw_center_x + fx * s_offset + out_x * self.lat
        self.y = self.rw_center_y + fy * s_offset + out_y * self.lat
        self.z = self.alt
        base_h = math.atan2(fy, fx)
        self.heading = base_h + math.radians(self.yaw)

    def update(self, dt):
        self.timer += dt
        if self.state == 'approach':
            self.s += self.speed * self.dir * dt
            denom = max(1.0, self.START_S - self.TOUCHDOWN_S)
            t = (self.START_S - self.s) / denom
            t = max(0.0, min(1.0, t))
            self.alt = self.START_ALT + (self.GROUND_ALT - self.START_ALT) * t
            if t > 0.86:
                flare = (t - 0.86) / 0.14
                self.pitch = -6.0 + 9.5 * flare
            else:
                self.pitch = -6.0
            if t >= 1.0 or self.s <= self.TOUCHDOWN_S:
                self.alt = self.GROUND_ALT
                self.pitch = 0.0
                self.state = 'rollout'
                self.timer = 0.0

        elif self.state == 'rollout':
            self.speed = max(0.0, self.speed - 7.5 * dt)
            self.s += self.speed * self.dir * dt
            self.pitch = self.pitch * max(0.0, 1.0 - 4.0 * dt)
            if self.speed <= 0.2:
                self.speed = 0.0
                self.state = 'turn'
                self.timer = 0.0

        elif self.state == 'turn':
            turn_dur = 3.5
            frac = min(1.0, self.timer / turn_dur)
            self.yaw = 180.0 + (180.0 * frac)
            if self.timer >= turn_dur:
                self.yaw = 0.0
                self.dir = 1.0
                self.state = 'wait'
                self.timer = 0.0

        elif self.state == 'wait':
            if self.timer >= 3.0:
                self.state = 'takeoff'
                self.timer = 0.0
                self.speed = 0.0

        elif self.state == 'takeoff':
            self.speed += 11.0 * dt
            self.s += self.speed * self.dir * dt
            if self.speed > 35.0:
                self.pitch = min(14.0, self.pitch + 4.5 * dt)
                climb = self.speed * math.sin(math.radians(max(0.0, self.pitch)))
                self.alt += climb * dt
            if self.alt > 18.0:
                self.gear_down = False
            if self.s > self.START_S + 180.0 or self.alt > 95.0:
                self.state = 'gone'
                self.timer = 0.0
                self.visible = False

        elif self.state == 'gone':
            self.visible = False
            if self.timer >= 4.0:
                self.reset()

        self._sync_world()


# =============================================================================
# RENDERER
# =============================================================================
class Renderer:
    def __init__(self, city, width, height):
        self.city = city
        self.W, self.H = width, height
        self.hw = width * 0.5
        self.f = Config.FOCAL_LENGTH
        self.pal_name = "DAY"
        self.pal = PALETTES["DAY"]
        self._lit_cache = {}
        self._glow = {}
        self._backdrops = {}
        self.surf = None
        self.td_scale = 5.0
        self.cam_mode = "FIRST_PERSON"
        self._poly_count = 0
        self._zo = 0.0
        self.hor = Config.HORIZON_Y
        self.cam_h = 1.3
        self.set_palette("DAY")
        if pygame.font.get_init():
            self.font_small = pygame.font.Font(None, 16)
            self.font_main = pygame.font.Font(None, 22)
        else:
            self.font_small = None
            self.font_main = None

    # ------------------------------------------------------------- palette
    def set_palette(self, name):
        self.pal_name = name
        self.pal = PALETTES[name]
        self._lit_cache = {}
        if name not in self._backdrops:
            self._backdrops[name] = self._make_backdrop(name)
        self.backdrop = self._backdrops[name]

    def _lit(self, col, d):
        pal = self.pal
        f = (d - pal['fog0']) / (pal['fog1'] - pal['fog0'])
        b = 0 if f <= 0.0 else (20 if f >= 1.0 else int(f * 20))
        key = (col, b)
        c = self._lit_cache.get(key)
        if c is None:
            t = pal['tint']
            base = (int(col[0] * t[0]), int(col[1] * t[1]), int(col[2] * t[2]))
            c = mix(base, pal['fog'], b / 20.0)
            self._lit_cache[key] = c
        return c

    def _make_backdrop(self, name):
        pal = PALETTES[name]
        Wb = int(2 * math.pi * Config.FOCAL_LENGTH)
        Hb = int(Config.HORIZON_Y + 40)
        s = pygame.Surface((Wb, Hb))
        hor = int(Config.HORIZON_Y)
        for y in range(hor + 1):
            s.fill(mix(pal['sky_top'], pal['sky_hor'], (y / hor) ** 1.3), (0, y, Wb, 1))
        rnd = random.Random(99)
        if pal['night']:
            for _ in range(260):
                x, y = rnd.randrange(Wb), rnd.randrange(int(hor * 0.9))
                b = rnd.randint(110, 230)
                s.set_at((x, y), (b, b, min(255, b + 20)))
        else:
            for _ in range(22):
                cx, cy = rnd.randrange(Wb), rnd.randrange(30, int(hor * 0.55))
                for _k in range(rnd.randint(4, 8)):
                    pygame.draw.ellipse(s, pal['cloud'], (cx + rnd.randint(-50, 50), cy + rnd.randint(-8, 8),
                                                          rnd.randint(60, 130), rnd.randint(18, 34)))
        sx = int(Wb * 0.62)
        sy = int(hor * (0.28 if name == "DAY" else (0.78 if name == "SUNSET" else 0.2)))
        pygame.draw.circle(s, pal['sun'], (sx, sy), 30 if name != "NIGHT" else 18)
        pts = [(0, hor)]
        for x in range(0, Wb + 12, 12):
            h = 30 + 18 * math.sin(x * 0.0041) + 12 * math.sin(x * 0.0113 + 1.3) + 7 * math.sin(x * 0.031)
            pts.append((x, hor - h))
        pts.append((Wb, hor))
        pygame.draw.polygon(s, pal['hills'], pts)
        pygame.draw.rect(s, pal['fog'], (0, hor - 3, Wb, Hb - hor + 3))
        return s

    def _glow_sprite(self, r, col):
        r = int(max(3, min(r, 90)))
        key = (r // 2, col)
        g = self._glow.get(key)
        if g is None:
            r = (r // 2) * 2 + 2
            g = pygame.Surface((r * 2, r * 2))
            g.fill((0, 0, 0))
            for i in range(r, 0, -2):
                k = (1.0 - i / r) ** 2
                pygame.draw.circle(g, (int(col[0] * k), int(col[1] * k), int(col[2] * k)), (r, r), i)
            self._glow[key] = g
        return g

    # ------------------------------------------------------------- camera
    def _set_cam(self, x, y, yaw, h, horizon):
        self.cam_x, self.cam_y = x, y
        self.cs, self.sn = math.cos(yaw), math.sin(yaw)
        self.cam_h = h
        self.hor = horizon
        self.yaw = yaw

    def _to_cam(self, wx, wy):
        dx, dy = wx - self.cam_x, wy - self.cam_y
        return -dx * self.sn + dy * self.cs, dx * self.cs + dy * self.sn

    @staticmethod
    def _clip_poly_2d(pts, x0, y0, x1, y1):
        """Sutherland-Hodgman polygon clipping against screen margin rect."""
        out = pts
        for edge in range(4):
            inp = out
            out = []
            if not inp:
                return []
            n = len(inp)
            for i in range(n):
                p1 = inp[i]
                p2 = inp[(i + 1) % n]
                if edge == 0:
                    in1, in2 = p1[0] >= x0, p2[0] >= x0
                    if in1:
                        out.append(p1)
                    if in1 != in2:
                        dx = p2[0] - p1[0]
                        if dx != 0.0:
                            out.append((x0, p1[1] + ((x0 - p1[0]) / dx) * (p2[1] - p1[1])))
                elif edge == 1:
                    in1, in2 = p1[0] <= x1, p2[0] <= x1
                    if in1:
                        out.append(p1)
                    if in1 != in2:
                        dx = p2[0] - p1[0]
                        if dx != 0.0:
                            out.append((x1, p1[1] + ((x1 - p1[0]) / dx) * (p2[1] - p1[1])))
                elif edge == 2:
                    in1, in2 = p1[1] >= y0, p2[1] >= y0
                    if in1:
                        out.append(p1)
                    if in1 != in2:
                        dy = p2[1] - p1[1]
                        if dy != 0.0:
                            out.append((p1[0] + ((y0 - p1[1]) / dy) * (p2[0] - p1[0]), y0))
                elif edge == 3:
                    in1, in2 = p1[1] <= y1, p2[1] <= y1
                    if in1:
                        out.append(p1)
                    if in1 != in2:
                        dy = p2[1] - p1[1]
                        if dy != 0.0:
                            out.append((p1[0] + ((y1 - p1[1]) / dy) * (p2[0] - p1[0]), y1))
        return out

    # --------------------------------------------------- polygon primitives
    def _poly_ground(self, poly, col):
        cx, cy, cs, sn = self.cam_x, self.cam_y, self.cs, self.sn
        near = Config.Z_NEAR
        vs = []
        mn = 1e9
        mx = -1e9
        for (wx, wy) in poly:
            dx, dy = wx - cx, wy - cy
            z = dx * cs + dy * sn
            vs.append((-dx * sn + dy * cs, z))
            if z < mn:
                mn = z
            if z > mx:
                mx = z
        if mx < near:
            return
        if mn < near:
            out = []
            n = len(vs)
            for i in range(n):
                a, b = vs[i], vs[(i + 1) % n]
                ai, bi = a[1] >= near, b[1] >= near
                if ai:
                    out.append(a)
                if ai != bi:
                    t = (near - a[1]) / (b[1] - a[1])
                    out.append((a[0] + (b[0] - a[0]) * t, near))
            vs = out
            if len(vs) < 3:
                return
        f, hw, hor, ch = self.f, self.hw, self.hor, self.cam_h
        pts = []
        W, H = self.W, self.H
        xmin = ymin = 1e9
        xmax = ymax = -1e9
        for (x, z) in vs:
            k = f / z
            sx = hw + x * k
            sy = hor + ch * k
            if sx < xmin:
                xmin = sx
            if sx > xmax:
                xmax = sx
            if sy < ymin:
                ymin = sy
            if sy > ymax:
                ymax = sy
            pts.append((sx, sy))
        if xmax < -50.0 or xmin > W + 50.0 or ymax < -50.0 or ymin > H + 50.0 or (xmax - xmin < 0.3 and ymax - ymin < 0.3):
            return
        if xmin < -50.0 or xmax > W + 50.0 or ymin < -50.0 or ymax > H + 50.0:
            clipped = self._clip_poly_2d(pts, -60.0, -60.0, W + 60.0, H + 60.0)
            if len(clipped) >= 3:
                pts = clipped
        pygame.draw.polygon(self.surf, col, pts)

    def _quad3(self, pts, col, outline=None):
        """General 3D polygon given world (x, y, z_height) vertices."""
        cx, cy, cs, sn = self.cam_x, self.cam_y, self.cs, self.sn
        near = Config.Z_NEAR
        zo = self._zo
        vs = []
        mx = -1e9
        mn = 1e9
        for (wx, wy, wz) in pts:
            dx, dy = wx - cx, wy - cy
            z = dx * cs + dy * sn
            vs.append((-dx * sn + dy * cs, wz + zo, z))
            if z > mx:
                mx = z
            if z < mn:
                mn = z
        if mx < near:
            return False
        if mn < near:
            out = []
            n = len(vs)
            for i in range(n):
                a, b = vs[i], vs[(i + 1) % n]
                ai, bi = a[2] >= near, b[2] >= near
                if ai:
                    out.append(a)
                if ai != bi:
                    t = (near - a[2]) / (b[2] - a[2])
                    out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near))
            vs = out
            if len(vs) < 3:
                return False
        f, hw, hor, ch = self.f, self.hw, self.hor, self.cam_h
        sp = []
        W, H = self.W, self.H
        xmin = ymin = 1e9
        xmax = ymax = -1e9
        for (x, h, z) in vs:
            k = f / z
            sx = hw + x * k
            sy = hor + (ch - h) * k
            if sx < xmin:
                xmin = sx
            if sx > xmax:
                xmax = sx
            if sy < ymin:
                ymin = sy
            if sy > ymax:
                ymax = sy
            sp.append((sx, sy))
        if xmax < 0 or xmin > W or ymax < 0 or ymin > H:
            return False
        if xmin < -50.0 or xmax > W + 50.0 or ymin < -50.0 or ymax > H + 50.0:
            sp = self._clip_poly_2d(sp, -50.0, -50.0, W + 50.0, H + 50.0)
            if len(sp) < 3:
                return False
        pygame.draw.polygon(self.surf, col, sp)
        if outline is not None:
            pygame.draw.polygon(self.surf, outline, sp, 1)
        return True

    def _box(self, corners, z0, z1, side_cols, top_col, face_fn=None):
        """Vertical prism over polygon `corners` (z0..z1)."""
        n = len(corners)
        ccx = sum(p[0] for p in corners) / n
        ccy = sum(p[1] for p in corners) / n
        camx, camy = self.cam_x, self.cam_y
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            ex, ey = b[0] - a[0], b[1] - a[1]
            nx, ny = ey, -ex
            mx, my = (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
            if nx * (mx - ccx) + ny * (my - ccy) < 0:
                nx, ny = -nx, -ny
            if nx * (camx - mx) + ny * (camy - my) <= 0:
                continue
            col = side_cols[i] if isinstance(side_cols, list) else side_cols
            self._quad3([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)], col)
            if face_fn is not None:
                face_fn(i, a, b)
        if self.cam_h > z1 + self._zo and top_col is not None:
            self._quad3([(p[0], p[1], z1) for p in corners], top_col)

    # ------------------------------------------------------- scene objects
    def _draw_building(self, b, dist, t=0.0):
        if getattr(b, 'landmark', None) is not None:
            self._draw_landmark(b, dist, t)
            return
        pal = self.pal
        col = b.color
        H = b.h
        corners = b.poly
        shades = [0.86, 1.0, 0.72, 0.9]
        cols = [self._lit(shade(col, k), dist) for k in shades]
        night = pal['night']

        def faces(i, a, bb):
            if dist > 130.0 or b.wall:
                return
            ex, ey = bb[0] - a[0], bb[1] - a[1]
            L = math.hypot(ex, ey)
            floors = min(int(H / 3.3), 12)
            wcol = (60, 78, 100)
            for fl in range(floors):
                z0 = 1.6 + fl * 3.3
                z1 = z0 + 1.5
                if z1 > H - 0.5:
                    break
                lit_w = night and ((b.bid * 7 + fl * 13 + i * 5) % 5 < 3)
                c = (255, 214, 120) if lit_w else wcol
                c = self._lit(c, dist) if not lit_w else c
                nwin = max(2, int(L / 4.0))
                if dist > 70.0:
                    nwin = 1
                for wi in range(nwin):
                    u0 = (wi + 0.18) / nwin
                    u1 = (wi + 0.82) / nwin
                    ax, ay = a[0] + ex * u0, a[1] + ey * u0
                    bx, by = a[0] + ex * u1, a[1] + ey * u1
                    self._quad3([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], c)

        self._box(corners, 0.0, H, cols, self._lit(shade(col, 1.12), dist), faces)

    def _draw_billboard_sign(self, cx, cy, cz, w, h, facing_angle, title_text, sub_text, text_col, border_col, dist, t):
        """Renders an illuminated 3D billboard signboard with physical backing and sharp typography."""
        c, s = math.cos(facing_angle), math.sin(facing_angle)
        rx, ry = -s, c
        hw, hh = w * 0.5, h * 0.5
        p0 = (cx - rx * hw, cy - ry * hw, cz - hh)
        p1 = (cx + rx * hw, cy + ry * hw, cz - hh)
        p2 = (cx + rx * hw, cy + ry * hw, cz + hh)
        p3 = (cx - rx * hw, cy - ry * hw, cz + hh)

        facing = (self.cam_x - cx) * c + (self.cam_y - cy) * s
        if facing <= 0.0 or dist > 260.0:
            return

        lit = self._lit
        night = self.pal['night'] or self.pal_name == "SUNSET"
        bg_col = (14, 18, 26) if night else (24, 28, 36)
        self._quad3([p0, p1, p2, p3], lit(bg_col, dist), border_col if night else lit(border_col, dist))

        px, pz = self._to_cam(cx, cy)
        if pz < 1.5:
            return
        k = self.f / pz
        sx = self.hw + px * k
        sy = self.hor + (self.cam_h - cz) * k

        w_screen = int(w * k)
        h_screen = int(h * k)
        if w_screen < 18 or h_screen < 7 or sx < -100 or sx > self.W + 100:
            return

        border_w = max(1, int(0.08 * k))
        rect_sc = pygame.Rect(int(sx - w_screen * 0.5), int(sy - h_screen * 0.5), w_screen, h_screen)
        pygame.draw.rect(self.surf, border_col, rect_sc, border_w, border_radius=max(1, int(0.12 * k)))

        font_sz = int(clamp(h_screen * 0.62, 12, 42))
        font = pygame.font.Font(None, font_sz)
        t_surf = font.render(title_text, True, text_col)
        if t_surf.get_width() > w_screen * 0.92:
            sc_factor = (w_screen * 0.92) / t_surf.get_width()
            new_w = int(w_screen * 0.92)
            new_h = max(6, int(t_surf.get_height() * sc_factor))
            t_surf = pygame.transform.smoothscale(t_surf, (new_w, new_h))

        self.surf.blit(t_surf, (int(sx - t_surf.get_width() * 0.5), int(sy - t_surf.get_height() * 0.5)))

        if night and dist < 160.0:
            g = self._glow_sprite(max(6, int(0.6 * k)), tuple(int(v * 0.7) for v in border_col))
            self.surf.blit(g, (int(sx - g.get_width() * 0.5), int(sy - g.get_height() * 0.5)), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_airplane(self, plane, dist, t):
        if not getattr(plane, 'visible', True):
            return
        lit = self._lit
        night = self.pal['night'] or self.pal_name == 'SUNSET'

        yaw = plane.heading
        pitch_rad = math.radians(plane.pitch)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)

        # Aircraft local basis in world space:
        fx = cos_y * cos_p
        fy = sin_y * cos_p
        fz = sin_p

        rx = -sin_y
        ry = cos_y
        rz = 0.0

        ux = -cos_y * sin_p
        uy = -sin_y * sin_p
        uz = cos_p

        def W(lx, ly, lz):
            return (
                plane.x + rx * lx + fx * ly + ux * lz,
                plane.y + ry * lx + fy * ly + uy * lz,
                plane.z + rz * lx + fz * ly + uz * lz
            )

        col_white = lit((242, 245, 250), dist)
        col_white_shade = lit((210, 215, 222), dist)
        col_white_top = lit((252, 254, 255), dist)
        col_wing = lit((225, 230, 238), dist)
        col_wing_bottom = lit((175, 180, 190), dist)
        col_glass = lit((35, 140, 210), dist)
        col_glass_night = (255, 230, 140) if night else lit((40, 95, 170), dist)
        col_tail = lit((225, 30, 30), dist)
        col_engine = lit((95, 100, 110), dist)
        col_engine_in = (30, 32, 36)
        col_gear = lit((50, 52, 56), dist)

        # 1. Fuselage Body (octagonal tube)
        rw, rh = 2.0, 2.1
        oct_pts = [
            (rw * 0.92, rh * 0.38),
            (rw * 0.38, rh * 0.92),
            (-rw * 0.38, rh * 0.92),
            (-rw * 0.92, rh * 0.38),
            (-rw * 0.92, -rh * 0.38),
            (-rw * 0.38, -rh * 0.92),
            (rw * 0.38, -rh * 0.92),
            (rw * 0.92, -rh * 0.38),
        ]
        n_oct = len(oct_pts)
        for i in range(n_oct):
            p0 = oct_pts[i]
            p1 = oct_pts[(i + 1) % n_oct]
            c_face = col_white_top if i in (1, 2) else (col_white_shade if i in (5, 6) else col_white)
            self._quad3([W(p0[0], -12.0, p0[1]), W(p1[0], -12.0, p1[1]),
                         W(p1[0], 11.0, p1[1]), W(p0[0], 11.0, p0[1])], c_face)

        # 2. Cockpit nose cone (ly=11.0 to ly=14.5)
        nose_tip = W(0.0, 14.5, -0.2)
        for i in range(n_oct):
            p0 = oct_pts[i]
            p1 = oct_pts[(i + 1) % n_oct]
            c_nose = col_glass if i in (0, 1, 2, 3) and p0[1] > 0 else col_white
            self._quad3([W(p0[0], 11.0, p0[1]), W(p1[0], 11.0, p1[1]), nose_tip, nose_tip], c_nose)

        # Cockpit windshield band
        self._quad3([W(-1.4, 11.5, 0.8), W(1.4, 11.5, 0.8),
                     W(0.9, 13.2, 0.5), W(-0.9, 13.2, 0.5)], col_glass)

        # 3. Passenger windows ribbon (starboard & port)
        if dist < 260.0:
            self._quad3([W(rw * 0.95, -9.0, 0.4), W(rw * 0.95, 9.0, 0.4),
                         W(rw * 0.95, 9.0, 0.9), W(rw * 0.95, -9.0, 0.9)], col_glass_night)
            self._quad3([W(-rw * 0.95, -9.0, 0.4), W(-rw * 0.95, 9.0, 0.4),
                         W(-rw * 0.95, 9.0, 0.9), W(-rw * 0.95, -9.0, 0.9)], col_glass_night)

        # 4. Main Wings (span 36m: lx from -18.0 to 18.0)
        self._quad3([W(-2.0, 3.2, -0.4), W(-18.0, -2.0, 0.3),
                     W(-18.0, -4.5, 0.3), W(-2.0, -2.8, -0.4)], col_wing)
        self._quad3([W(-2.0, -2.8, -0.6), W(-18.0, -4.5, 0.1),
                     W(-18.0, -2.0, 0.1), W(-2.0, 3.2, -0.6)], col_wing_bottom)
        self._quad3([W(2.0, 3.2, -0.4), W(18.0, -2.0, 0.3),
                     W(18.0, -4.5, 0.3), W(2.0, -2.8, -0.4)], col_wing)
        self._quad3([W(2.0, -2.8, -0.6), W(18.0, -4.5, 0.1),
                     W(18.0, -2.0, 0.1), W(2.0, 3.2, -0.6)], col_wing_bottom)

        # Wingtip navigation lights & strobes
        pt_l_tip = W(-18.0, -3.2, 0.3)
        pt_r_tip = W(18.0, -3.2, 0.3)
        self._draw_beacon(pt_l_tip[0], pt_l_tip[1], pt_l_tip[2], (255, 30, 30), dist)
        self._draw_beacon(pt_r_tip[0], pt_r_tip[1], pt_r_tip[2], (30, 255, 80), dist)
        if int(t * 1.8) % 2 == 0:
            self._draw_beacon(pt_l_tip[0], pt_l_tip[1], pt_l_tip[2] + 0.15, (255, 255, 255), dist)
            self._draw_beacon(pt_r_tip[0], pt_r_tip[1], pt_r_tip[2] + 0.15, (255, 255, 255), dist)

        # 5. Tail Fin (height 7m, bold red)
        self._quad3([W(-0.3, -9.0, 2.0), W(0.3, -9.0, 2.0),
                     W(0.2, -13.0, 8.8), W(-0.2, -13.0, 8.8)], col_tail)
        self._quad3([W(-0.3, -12.8, 2.0), W(0.3, -12.8, 2.0),
                     W(0.2, -14.6, 8.6), W(-0.2, -14.6, 8.6)], col_tail)
        self._quad3([W(-0.25, -9.0, 2.0), W(-0.2, -13.0, 8.8),
                     W(-0.2, -14.6, 8.6), W(-0.25, -12.8, 2.0)], col_tail)
        self._quad3([W(0.25, -9.0, 2.0), W(0.2, -13.0, 8.8),
                     W(0.2, -14.6, 8.6), W(0.25, -12.8, 2.0)], col_tail)

        # 6. Horizontal Tail Stabilizers (span 12m)
        self._quad3([W(-6.0, -11.0, 1.2), W(-1.0, -10.0, 1.2),
                     W(-1.0, -13.2, 1.2), W(-6.0, -13.6, 1.2)], col_wing)
        self._quad3([W(1.0, -10.0, 1.2), W(6.0, -11.0, 1.2),
                     W(6.0, -13.6, 1.2), W(1.0, -13.2, 1.2)], col_wing)

        # 7. Jet Engines (2 engines at lx = +/- 7.5)
        for sx in (-7.5, 7.5):
            ey, ez = 1.0, -1.8
            ew, el = 1.1, 4.8
            self._quad3([W(sx - ew, ey - el*0.5, ez), W(sx + ew, ey - el*0.5, ez),
                         W(sx + ew, ey + el*0.5, ez), W(sx - ew, ey + el*0.5, ez)], col_engine)
            self._quad3([W(sx - ew, ey + el*0.5, ez - ew), W(sx + ew, ey + el*0.5, ez - ew),
                         W(sx + ew, ey + el*0.5, ez + ew), W(sx - ew, ey + el*0.5, ez + ew)], col_engine_in)
            if plane.state == 'takeoff':
                thrust_c = (255, 160, 40) if int(t * 12) % 2 == 0 else (100, 180, 255)
                self._draw_beacon(W(sx, ey - el*0.5 - 0.5, ez)[0],
                                  W(sx, ey - el*0.5 - 0.5, ez)[1],
                                  W(sx, ey - el*0.5 - 0.5, ez)[2], thrust_c, dist)

        # 8. Landing Gear (when gear_down is True)
        if plane.gear_down and dist < 220.0:
            for gx, gy in [(0.0, 8.5), (-3.5, -1.0), (3.5, -1.0)]:
                strut_top = W(gx, gy, -1.5)
                strut_bot = W(gx, gy, -3.2)
                self._quad3([W(gx - 0.25, gy, -1.5), W(gx + 0.25, gy, -1.5),
                             W(gx + 0.25, gy, -3.2), W(gx - 0.25, gy, -3.2)], col_gear)
                wheel_w = 0.5
                self._quad3([W(gx - wheel_w, gy - 0.4, -3.2), W(gx + wheel_w, gy - 0.4, -3.2),
                             W(gx + wheel_w, gy + 0.4, -3.2), W(gx - wheel_w, gy + 0.4, -3.2)], (25, 25, 28))

        # 9. Anticollision Red Beacon on dorsal ridge
        top_beacon = W(0.0, 1.0, 2.3)
        self._draw_beacon(top_beacon[0], top_beacon[1], top_beacon[2],
                          (255, 30, 30) if int(t * 3.0) % 2 == 0 else (120, 20, 20), dist)

    def _draw_landmark(self, b, dist, t):
        lm = b.landmark
        tag = lm['tag']
        lit = self._lit
        x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
        cx, cy = b.cx, b.cy
        H = b.h
        night = self.pal['night'] or self.pal_name == "SUNSET"
        col = b.color
        shades = [0.86, 1.0, 0.72, 0.9]
        cols = [lit(shade(col, k), dist) for k in shades]

        out_vec = lm.get('out_vec', (0.0, 1.0))
        f_vec = lm.get('f_vec', (1.0, 0.0))
        out_x, out_y = out_vec
        fx, fy = f_vec
        bw = lm.get('bw', x1 - x0)
        bd = lm.get('bd', y1 - y0)
        facing_ang = math.atan2(-out_y, -out_x)

        # 1. Complete, solid, properly built building (for ALL landmarks)
        def faces(i, a, bb):
            if dist > 140.0:
                return
            ex, ey = bb[0] - a[0], bb[1] - a[1]
            L = math.hypot(ex, ey)
            floors = min(int(H / 3.4), 14)
            wcol = (60, 80, 105)
            for fl in range(floors):
                z0 = 1.8 + fl * 3.4
                z1 = z0 + 1.6
                if z1 > H - 0.6:
                    break
                lit_w = night and ((b.bid * 7 + fl * 13 + i * 5) % 5 < 3)
                c = (255, 220, 130) if lit_w else wcol
                c = self._lit(c, dist) if not lit_w else c
                nwin = max(2, int(L / 4.2))
                for wi in range(nwin):
                    u0 = (wi + 0.15) / nwin
                    u1 = (wi + 0.85) / nwin
                    ax, ay = a[0] + ex * u0, a[1] + ey * u0
                    bx, by = a[0] + ex * u1, a[1] + ey * u1
                    self._quad3([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], c)

        self._box(b.poly, 0.0, H, cols, lit(shade(col, 1.12), dist), faces)

        # 2. Simple, bold signboard on building facade with name of the destination
        b_front_x = cx - out_x * (bd * 0.50 + 0.3)
        b_front_y = cy - out_y * (bd * 0.50 + 0.3)
        sign_w = min(bw * 0.70, 28.0)
        sign_h = 4.2
        sign_z = min(H * 0.75, 12.0)
        if H < 15.0:
            sign_z = H + 2.5

        sign_titles = {
            'airport': "AIRPORT",
            'hospital': "HOSPITAL",
            'mall': "SHOPPING MALL",
            'skyscraper': "SKYSCRAPER TOWER",
            'market': "CENTRAL MARKET",
            'bus_station': "BUS TERMINUS",
            'railway': "RAILWAY STATION"
        }
        title = sign_titles.get(tag, lm.get('name', 'DESTINATION'))
        self._draw_billboard_sign(b_front_x, b_front_y, sign_z, sign_w, sign_h,
                                 facing_ang, title, "", (255, 255, 255),
                                 lm['color'], dist, t)

        # 3. Simple, clean entrance gateposts at the street curb
        gp1, gp2 = lm.get('gate_p1'), lm.get('gate_p2')
        if gp1 and gp2 and dist < 220.0:
            pw, gh = 1.2, 4.5
            for gp in (gp1, gp2):
                poly_gp = [(gp[0] - pw*0.5, gp[1] - pw*0.5), (gp[0] + pw*0.5, gp[1] - pw*0.5),
                           (gp[0] + pw*0.5, gp[1] + pw*0.5), (gp[0] - pw*0.5, gp[1] + pw*0.5)]
                self._box(poly_gp, 0.0, gh, [lit((60, 64, 72), dist)] * 4, lit((80, 85, 95), dist))
                pass # Ground gatepost beacon removed

        # 4. Drop-off zone (the goal) with a tall light beam visible from far away
        fp = lm.get('finish_pt')
        fb = lm.get('finish_box')
        if fp:
            pass # Vertical green glow beam removed - ground parking box and bollards preserved

        if fp and fb and dist < 220.0:
            border_col = (80, 255, 140) if int(t * 3.0) % 2 == 0 else (245, 215, 40)
            for bi in range(4):
                pa, pb = fb[bi], fb[(bi + 1) % 4]
                self._quad3([(pa[0], pa[1], 0.03), (pb[0], pb[1], 0.03),
                             (pb[0] * 0.94 + fp[0] * 0.06, pb[1] * 0.94 + fp[1] * 0.06, 0.03),
                             (pa[0] * 0.94 + fp[0] * 0.06, pa[1] * 0.94 + fp[1] * 0.06, 0.03)], border_col)
            # Ground arrival directional chevron arrow
            ax = (fb[0][0] + fb[1][0]) * 0.5
            ay = (fb[0][1] + fb[1][1]) * 0.5
            tip_x, tip_y = ax + out_x * 2.5, ay + out_y * 2.5
            self._quad3([(ax - fx * 1.5, ay - fy * 1.5, 0.04), (tip_x, tip_y, 0.04),
                         (tip_x, tip_y, 0.04), (ax + fx * 1.5, ay + fy * 1.5, 0.04)], (250, 210, 30))
            # 4 grounded low corner bollards (height 0.9m)
            f_bw, f_bd = 3.6, 4.2
            for sign_x in (-1, 1):
                for sign_y in (-1, 1):
                    bx_pt = fp[0] + fx * f_bw * sign_x + out_x * f_bd * sign_y
                    by_pt = fp[1] + fy * f_bw * sign_x + out_y * f_bd * sign_y
                    poly_b = [(bx_pt - 0.25, by_pt - 0.25), (bx_pt + 0.25, by_pt - 0.25),
                              (bx_pt + 0.25, by_pt + 0.25), (bx_pt - 0.25, by_pt + 0.25)]
                    self._box(poly_b, 0.0, 0.9, [lit((70, 75, 85), dist)] * 4, (240, 210, 40))
                    pass # Bollard beacon removed

        # 5. Clean, grounded architectural identity per landmark
        if tag == 'airport':
            # Terminal Building: Continuous cyan glass ribbon band at mid-height
            self._quad3([(b_front_x - fx * (bw * 0.48), b_front_y - fy * (bw * 0.48), 5.5),
                         (b_front_x + fx * (bw * 0.48), b_front_y + fy * (bw * 0.48), 5.5),
                         (b_front_x + fx * (bw * 0.48), b_front_y + fy * (bw * 0.48), 10.5),
                         (b_front_x - fx * (bw * 0.48), b_front_y - fy * (bw * 0.48), 10.5)], lit((50, 165, 225), dist))
            # Terminal cantilevered roof overhang slab (gray 0.4)
            roof_slab = [
                (cx - fx * (bw * 0.52) - out_x * (bd * 0.52), cy - fy * (bw * 0.52) - out_y * (bd * 0.52)),
                (cx + fx * (bw * 0.52) - out_x * (bd * 0.52), cy + fy * (bw * 0.52) - out_y * (bd * 0.52)),
                (cx + fx * (bw * 0.52) + out_x * (bd * 0.52), cy + fy * (bw * 0.52) + out_y * (bd * 0.52)),
                (cx - fx * (bw * 0.52) + out_x * (bd * 0.52), cy - fy * (bw * 0.52) + out_y * (bd * 0.52))
            ]
            self._box(roof_slab, H, H + 1.2, [lit((90, 95, 102), dist)] * 4, lit((110, 115, 122), dist))

            # Air Traffic Control Tower: 40m column, observation cab, and flashing beacon
            tw_x = cx + fx * (bw * 0.48) - out_x * 8.0
            tw_y = cy + fy * (bw * 0.48) - out_y * 8.0
            tw_col = [(tw_x - 3.0, tw_y - 3.0), (tw_x + 3.0, tw_y - 3.0),
                      (tw_x + 3.0, tw_y + 3.0), (tw_x - 3.0, tw_y + 3.0)]
            self._box(tw_col, 0.0, 38.0, [lit((170, 175, 185), dist)] * 4, lit((190, 195, 205), dist))
            tw_cab = [(tw_x - 7.0, tw_y - 7.0), (tw_x + 7.0, tw_y - 7.0),
                      (tw_x + 7.0, tw_y + 7.0), (tw_x - 7.0, tw_y + 7.0)]
            self._box(tw_cab, 38.0, 44.0, [lit((45, 170, 230), dist)] * 4, lit((80, 85, 95), dist))
            tw_roof = [(tw_x - 8.0, tw_y - 8.0), (tw_x + 8.0, tw_y - 8.0),
                       (tw_x + 8.0, tw_y + 8.0), (tw_x - 8.0, tw_y + 8.0)]
            self._box(tw_roof, 44.0, 45.2, [lit((70, 75, 82), dist)] * 4, lit((85, 90, 98), dist))
            self._draw_beacon(tw_x, tw_y, 46.5, (255, 30, 30) if int(t * 2.5) % 2 == 0 else (120, 20, 20), dist)

            # Airport grounds stay clean and within airport boundaries
        elif tag == 'hospital':
            # Large red medical cross on facade
            if dist < 180.0:
                rc_col = (235, 30, 30)
                zc = sign_z + 4.5 if sign_z + 4.5 < H - 2.0 else H * 0.65
                self._quad3([(b_front_x - fx * 0.8, b_front_y - fy * 0.8, zc - 2.8),
                             (b_front_x + fx * 0.8, b_front_y + fy * 0.8, zc - 2.8),
                             (b_front_x + fx * 0.8, b_front_y + fy * 0.8, zc + 2.8),
                             (b_front_x - fx * 0.8, b_front_y - fy * 0.8, zc + 2.8)], rc_col)
                self._quad3([(b_front_x - fx * 2.8, b_front_y - fy * 2.8, zc - 0.8),
                             (b_front_x + fx * 2.8, b_front_y + fy * 2.8, zc - 0.8),
                             (b_front_x + fx * 2.8, b_front_y + fy * 2.8, zc + 0.8),
                             (b_front_x - fx * 2.8, b_front_y - fy * 2.8, zc + 0.8)], rc_col)
            # Rooftop helipad
            pad_hw = 7.0
            pad = [(cx - pad_hw, cy - pad_hw, H + 0.04), (cx + pad_hw, cy - pad_hw, H + 0.04),
                   (cx + pad_hw, cy + pad_hw, H + 0.04), (cx - pad_hw, cy + pad_hw, H + 0.04)]
            self._quad3(pad, lit((90, 92, 98), dist))
            hy_c = lit((250, 215, 35), dist)
            for ox in (-2.4, 2.4):
                self._quad3([(cx + ox - 0.45, cy - 3.2, H + 0.06), (cx + ox + 0.45, cy - 3.2, H + 0.06),
                             (cx + ox + 0.45, cy + 3.2, H + 0.06), (cx + ox - 0.45, cy + 3.2, H + 0.06)], hy_c)
            self._quad3([(cx - 2.4, cy - 0.45, H + 0.06), (cx + 2.4, cy - 0.45, H + 0.06),
                         (cx + 2.4, cy + 0.45, H + 0.06), (cx - 2.4, cy + 0.45, H + 0.06)], hy_c)
            for sx_c in (-1, 1):
                for sy_c in (-1, 1):
                    self._draw_beacon(cx + sx_c * pad_hw * 0.9, cy + sy_c * pad_hw * 0.9, H + 0.2,
                                      (255, 40, 30) if int(t * 2) % 2 == 0 else (255, 230, 40), dist)

        elif tag == 'skyscraper':
            # Stepped towers
            in1 = 3.5
            p1 = [(x0 + in1, y0 + in1), (x1 - in1, y0 + in1), (x1 - in1, y1 - in1), (x0 + in1, y1 - in1)]
            self._box(p1, H, H + 20.0, [lit(shade((75, 120, 175), k), dist) for k in (0.86, 1.0, 0.72, 0.9)],
                      lit((85, 135, 195), dist))
            in2 = 7.0
            p2 = [(x0 + in2, y0 + in2), (x1 - in2, y0 + in2), (x1 - in2, y1 - in2), (x0 + in2, y1 - in2)]
            self._box(p2, H + 20.0, H + 36.0, [lit(shade((90, 140, 200), k), dist) for k in (0.86, 1.0, 0.72, 0.9)],
                      lit((100, 155, 215), dist))
            spire_base = H + 36.0
            spire_top = spire_base + 14.0
            self._quad3([(cx - 0.4, cy, spire_base), (cx + 0.4, cy, spire_base),
                 (cx + 0.4, cy, spire_top), (cx - 0.4, cy, spire_top)], lit((160, 165, 175), dist))
            self._draw_beacon(cx, cy, spire_top, (255, 30, 20), dist)

        elif tag == 'mall':
            # Entrance colonnade canopy
            can_w, can_d, can_h = min(bw * 0.5, 18.0), 4.0, 4.5
            mc_x = b_front_x - out_x * (can_d * 0.5)
            mc_y = b_front_y - out_y * (can_d * 0.5)
            can_p = [(mc_x - fx * can_w*0.5, mc_y - fy * can_w*0.5, can_h),
                     (mc_x + fx * can_w*0.5, mc_y + fy * can_w*0.5, can_h),
                     (mc_x + fx * can_w*0.5 + out_x * can_d, mc_y + fy * can_w*0.5 + out_y * can_d, can_h),
                     (mc_x - fx * can_w*0.5 + out_x * can_d, mc_y - fy * can_w*0.5 + out_y * can_d, can_h)]
            self._quad3(can_p, lit((235, 195, 45), dist))
            # Storefront windows
            if dist < 120.0:
                for wi in range(4):
                    wx_off = -can_w * 0.38 + wi * (can_w * 0.25)
                    w_pos_x = b_front_x + fx * wx_off
                    w_pos_y = b_front_y + fy * wx_off
                    store_col = [(255, 160, 60), (80, 200, 255), (255, 90, 180), (100, 255, 140)][wi % 4]
                    self._quad3([(w_pos_x - fx * 1.4, w_pos_y - fy * 1.4, 1.0),
                                 (w_pos_x + fx * 1.4, w_pos_y + fy * 1.4, 1.0),
                                 (w_pos_x + fx * 1.4, w_pos_y + fy * 1.4, 3.6),
                                 (w_pos_x - fx * 1.4, w_pos_y - fy * 1.4, 3.6)], lit(store_col, dist))

        elif tag == 'market':
            can_w, can_d, can_h = 14.0, 3.5, 4.2
            mc_x = b_front_x - out_x * (can_d * 0.5)
            mc_y = b_front_y - out_y * (can_d * 0.5)
            can_p = [(mc_x - fx * can_w*0.5, mc_y - fy * can_w*0.5, can_h),
                     (mc_x + fx * can_w*0.5, mc_y + fy * can_w*0.5, can_h),
                     (mc_x + fx * can_w*0.5 + out_x * can_d, mc_y + fy * can_w*0.5 + out_y * can_d, can_h),
                     (mc_x - fx * can_w*0.5 + out_x * can_d, mc_y - fy * can_w*0.5 + out_y * can_d, can_h)]
            self._quad3(can_p, lit((210, 80, 40), dist))

        elif tag == 'bus_station':
            can_w, can_d, can_h = min(bw * 0.6, 22.0), 4.5, 4.4
            mc_x = b_front_x - out_x * (can_d * 0.5)
            mc_y = b_front_y - out_y * (can_d * 0.5)
            can_p = [(mc_x - fx * can_w*0.5, mc_y - fy * can_w*0.5, can_h),
                     (mc_x + fx * can_w*0.5, mc_y + fy * can_w*0.5, can_h),
                     (mc_x + fx * can_w*0.5 + out_x * can_d, mc_y + fy * can_w*0.5 + out_y * can_d, can_h),
                     (mc_x - fx * can_w*0.5 + out_x * can_d, mc_y - fy * can_w*0.5 + out_y * can_d, can_h)]
            self._quad3(can_p, lit((160, 165, 175), dist))

        elif tag == 'railway':
            tw_h = H + 8.0
            tw_w = 6.0
            tw_x = cx - fx * (bw * 0.3)
            tw_y = cy - fy * (bw * 0.3)
            tw_poly = [(tw_x - tw_w*0.5, tw_y - tw_w*0.5), (tw_x + tw_w*0.5, tw_y - tw_w*0.5),
                       (tw_x + tw_w*0.5, tw_y + tw_w*0.5), (tw_x - tw_w*0.5, tw_y + tw_w*0.5)]
            self._box(tw_poly, H, tw_h, [lit(shade((165, 80, 65), k), dist) for k in (0.86, 1.0, 0.72, 0.9)],
                      lit((195, 100, 80), dist))
            cy_cf = tw_h - 2.5
            c_fx = tw_x - out_x * (tw_w * 0.5 + 0.1)
            c_fy = tw_y - out_y * (tw_w * 0.5 + 0.1)
            self._quad3([(c_fx - fx * 1.5, c_fy - fy * 1.5, cy_cf - 1.5),
                         (c_fx + fx * 1.5, c_fy + fy * 1.5, cy_cf - 1.5),
                         (c_fx + fx * 1.5, c_fy + fy * 1.5, cy_cf + 1.5),
                         (c_fx - fx * 1.5, c_fy - fy * 1.5, cy_cf + 1.5)], (255, 250, 235))
            self._draw_beacon(c_fx, c_fy, cy_cf, (255, 240, 200), dist)

    def _draw_beacon(self, x, y, z, col, dist):
        px, pz = self._to_cam(x, y)
        if pz < 0.6:
            return
        k = self.f / pz
        sx = self.hw + px * k
        sy = self.hor + (self.cam_h - z) * k
        r = max(2, int(0.4 * k))
        pygame.draw.circle(self.surf, col, (int(sx), int(sy)), r)
        if dist < 260.0:
            g = self._glow_sprite(max(4, 0.8 * k), col)
            self.surf.blit(g, (int(sx) - g.get_width() // 2, int(sy) - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_tree(self, t, dist):
        x, y, h, rv = t
        px, pz = self._to_cam(x, y)
        if pz < 0.6:
            return
        k = self.f / pz
        sx = self.hw + px * k
        base = self.hor + self.cam_h * k
        top = self.hor + (self.cam_h - h) * k
        if sx < -200 or sx > self.W + 200:
            return
        trunk = self._lit((86, 62, 40), dist)
        th = (base - top) * 0.45
        pygame.draw.line(self.surf, trunk, (sx, base), (sx, base - th), max(1, int(0.32 * k)))
        green = [(28, 105, 38), (34, 125, 46), (22, 85, 30), (46, 140, 55), (60, 130, 40)][int(rv * 5) % 5]
        r = max(2, int(h * 0.30 * k))
        cy0 = base - th - r * 0.4
        for i, (dx, dy, rr) in enumerate(((0, 0, 1.0), (-0.45, 0.35, 0.72), (0.45, 0.30, 0.72), (0, -0.5, 0.7))):
            c = self._lit(shade(green, 0.85 + 0.1 * i), dist)
            pygame.draw.circle(self.surf, c, (int(sx + dx * r), int(cy0 + dy * r)), max(1, int(r * rr)))

    def _draw_lamp(self, l, dist):
        x, y, h, side, z = l
        player = getattr(getattr(self, 'sim', None), 'player', None)
        # Suppress ground lamps that sit under or next to flyover decks when player/camera is elevated
        if (self.cam_h > 2.0 or (player and (player.z > 1.2 or getattr(player, 'on_bridge', False)))) and z < 2.0:
            if self.city.deck_z(x, y) is not None or self.city.deck_hash.query_box(x - 8, y - 8, x + 8, y + 8):
                return
        px, pz = self._to_cam(x, y)
        if pz < 0.6:
            return
        k = self.f / pz
        sx = self.hw + px * k
        base = self.hor + (self.cam_h - z) * k
        top = self.hor + (self.cam_h - z - h) * k
        if sx < -100 or sx > self.W + 100:
            return
        c = self._lit((90, 94, 100), dist)
        pygame.draw.line(self.surf, c, (sx, base), (sx, top), max(1, int(0.18 * k)))
        pygame.draw.circle(self.surf, self._lit((230, 230, 200), dist) if not self.pal['night'] else (255, 244, 190),
                           (int(sx), int(top)), max(1, int(0.45 * k)))
        if self.pal['glow'] > 0.0 and dist < 160.0:
            g = self._glow_sprite(0.5 * k * (1.0 + self.pal['glow']) + 6, (170, 150, 90))
            self.surf.blit(g, (sx - g.get_width() // 2, top - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)


    def _draw_sign(self, s, dist):
        x, y, z, h = s['x'], s['y'], s.get('z', 0.0), s.get('h', 2.7)
        player = getattr(getattr(self, 'sim', None), 'player', None)
        # If camera is elevated/on flyover and sign is at ground level underneath the deck, skip it
        if (self.cam_h > 2.0 or (player and (player.z > 1.2 or getattr(player, 'on_bridge', False)))) and z < 1.0:
            if self.city.deck_z(x, y) is not None or self.city.deck_hash.query_box(x - 8, y - 8, x + 8, y + 8):
                return
        px, pz = self._to_cam(x, y)
        if pz < 0.6:
            return
        k = self.f / pz
        sx = self.hw + px * k
        base = self.hor + (self.cam_h - z) * k
        top = self.hor + (self.cam_h - z - h) * k
        if sx < -100 or sx > self.W + 100:
            return

        # 1. Pole (metallic grey)
        pole_c = self._lit((130, 135, 142), dist)
        pygame.draw.line(self.surf, pole_c, (int(sx), int(base)), (int(sx), int(top)), max(1, int(0.12 * k)))

        # 2. Sign face
        kind = s.get('kind', 'speed')
        val = s.get('val', 40)
        sign_r = max(3, int(0.48 * k))

        if kind == 'speed':
            # Circular Speed Limit Sign: Red outer ring, white circular background, black number
            c_red = self._lit((225, 35, 35), dist)
            c_white = self._lit((250, 250, 250), dist)
            c_black = self._lit((20, 20, 25), dist)
            pygame.draw.circle(self.surf, c_red, (int(sx), int(top)), sign_r)
            pygame.draw.circle(self.surf, c_white, (int(sx), int(top)), max(1, int(sign_r * 0.78)))
            if dist < 45.0 and sign_r >= 7:
                if not getattr(self, 'font_small', None) and pygame.font.get_init():
                    self.font_small = pygame.font.Font(None, 16)
                    self.font_main = pygame.font.Font(None, 22)
                fnt = self.font_small if sign_r < 13 else getattr(self, 'font_main', None)
                if fnt is not None:
                    t_lbl = fnt.render(str(val), True, c_black)
                    self.surf.blit(t_lbl, (int(sx - t_lbl.get_width() * 0.5), int(top - t_lbl.get_height() * 0.5)))
        elif kind == 'zebra':
            # Blue square sign with white walking pedestrian / stripes
            c_blue = self._lit((25, 95, 220), dist)
            c_white = self._lit((245, 245, 250), dist)
            rect_w = int(sign_r * 1.8)
            rect_h = int(sign_r * 1.8)
            rx0 = int(sx - rect_w * 0.5)
            ry0 = int(top - rect_h * 0.5)
            pygame.draw.rect(self.surf, c_blue, (rx0, ry0, rect_w, rect_h))
            pygame.draw.rect(self.surf, c_white, (rx0, ry0, rect_w, rect_h), max(1, int(sign_r * 0.12)))
            if dist < 45.0 and sign_r >= 6:
                pygame.draw.line(self.surf, c_white, (rx0 + 2, ry0 + rect_h - 4), (rx0 + rect_w - 2, ry0 + rect_h - 4), max(1, int(sign_r * 0.2)))
                pygame.draw.line(self.surf, c_white, (int(sx), ry0 + 4), (int(sx), ry0 + rect_h - 6), max(1, int(sign_r * 0.2)))
        elif kind == 'stop':
            c_red = self._lit((210, 25, 25), dist)
            c_white = self._lit((250, 250, 250), dist)
            pygame.draw.circle(self.surf, c_red, (int(sx), int(top)), sign_r)
            pygame.draw.circle(self.surf, c_white, (int(sx), int(top)), sign_r, max(1, int(sign_r * 0.12)))
            if dist < 45.0 and sign_r >= 8:
                if not getattr(self, 'font_small', None) and pygame.font.get_init():
                    self.font_small = pygame.font.Font(None, 16)
                if getattr(self, 'font_small', None) is not None:
                    t_lbl = self.font_small.render("STOP", True, c_white)
                    self.surf.blit(t_lbl, (int(sx - t_lbl.get_width() * 0.5), int(top - t_lbl.get_height() * 0.5)))

    def _draw_signal(self, hd, dist, t):
        x, y = hd['x'], hd['y']
        dot_cam = (self.cam_x - x) * hd['fx'] + (self.cam_y - y) * hd['fy']
        align = hd['fx'] * self.cs + hd['fy'] * self.sn
        if dot_cam <= 0.0 or align > -0.25:
            return
        best = getattr(self, 'sig_best_axis', {}).get(hd['sig'])
        if best is not None and hd['axis'] != best[1]:
            return
        px, pz = self._to_cam(x, y)
        if pz < 0.6:
            return
        k = self.f / pz
        sx = self.hw + px * k
        base = self.hor + self.cam_h * k
        top = self.hor + (self.cam_h - 5.8) * k
        if sx < -100 or sx > self.W + 100:
            return
        pole = self._lit((60, 62, 68), dist)
        pygame.draw.line(self.surf, pole, (sx, base), (sx, top), max(1, int(0.22 * k)))
        # Horizontal cantilever bracket holding signal head
        arm_w = int(0.40 * k)
        pygame.draw.line(self.surf, pole, (sx, top + 0.25 * k), (sx + arm_w, top + 0.25 * k), max(1, int(0.12 * k)))
        
        # Indian Signal Backboard: Black body with high-visibility bright yellow border (IRC:67 standard)
        bw, bh = max(4, int(0.60 * k)), max(10, int(1.7 * k))
        hx_center = sx + arm_w * 0.5
        rect = pygame.Rect(int(hx_center - bw / 2), int(top - 0.15 * k), bw, bh)
        pygame.draw.rect(self.surf, (16, 18, 22), rect, border_radius=max(1, bw // 4))
        border_w = max(1, int(0.06 * k))
        pygame.draw.rect(self.surf, (245, 205, 30), rect, width=border_w, border_radius=max(1, bw // 4))
        
        st = hd['sig'].state(hd['axis'], t)
        cols = {'R': (255, 35, 25), 'Y': (255, 185, 20), 'G': (25, 240, 95)}
        lamp_r = max(1, int(bw * 0.35))
        
        for i, key in enumerate(('R', 'Y', 'G')):
            cy_lamp = int(rect.y + bh * (0.20 + 0.30 * i))
            cc = cols[key] if st == key else (32, 34, 38)
            pygame.draw.circle(self.surf, cc, (int(hx_center), cy_lamp), lamp_r)
            visor_w = max(1, int(0.05 * k))
            pygame.draw.arc(self.surf, (15, 15, 18),
                            (int(hx_center - lamp_r - 1), int(cy_lamp - lamp_r - visor_w), (lamp_r + 1) * 2, (lamp_r + 1) * 2),
                            0.0, math.pi, visor_w)
            
            # When GREEN: render sharp Indian traffic directional straight arrow (↑) inside the green lens
            if key == 'G' and st == 'G' and dist < 140.0 and lamp_r >= 3:
                arr_h = max(2, int(lamp_r * 0.75))
                arr_th = max(1, int(lamp_r * 0.22))
                pygame.draw.line(self.surf, (10, 35, 15), (int(hx_center), cy_lamp - arr_h), (int(hx_center), cy_lamp + arr_h), arr_th)
                ah_w = max(2, int(lamp_r * 0.55))
                tip_y = cy_lamp - arr_h
                pygame.draw.polygon(self.surf, (10, 35, 15), [
                    (int(hx_center), tip_y - 1),
                    (int(hx_center - ah_w), tip_y + ah_w),
                    (int(hx_center + ah_w), tip_y + ah_w)
                ])

        if dist < 220.0:
            c = cols[st]
            g = self._glow_sprite(max(5, 0.9 * k), tuple(int(v * 0.8) for v in c))
            gy = rect.y + bh * (0.20 + 0.30 * ('R', 'Y', 'G').index(st))
            self.surf.blit(g, (int(hx_center) - g.get_width() // 2, int(gy) - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_barrier(self, bar, dist):
        poly = bar[7]
        h = bar[5]
        if bar[6] == 'median':
            side, top = (178, 176, 170), (200, 198, 192)
        else:
            side, top = (150, 154, 162), (185, 190, 198)
        self._box(poly, 0.0, h, self._lit(side, dist), self._lit(top, dist))

    # ---------------------------------------------------------- flyovers
    def _draw_deck(self, ch, dist):
        c0, c1 = ch['c0'], ch['c1']
        z0, z1 = ch['z0'], ch['z1']
        rx, ry = ch['r']
        hd, hw = ch['hd'], ch['hw']
        th = DECK_T
        lit = self._lit
        solid = ch['solid']
        mx, my = ch['cx'], ch['cy']
        zt = (z0 + z1) * 0.5
        camx, camy, camh = self.cam_x, self.cam_y, self.cam_h
        dx_c = c1[0] - c0[0]
        dy_c = c1[1] - c0[1]
        sz = z1 - z0
        seg_len_sq = dx_c * dx_c + dy_c * dy_c
        if seg_len_sq > 0.01:
            proj = clamp(((camx - c0[0]) * dx_c + (camy - c0[1]) * dy_c) / seg_len_sq, 0.0, 1.0)
            z_plane = z0 + proj * sz
        else:
            z_plane = zt
        # Bridge climbing fix: when driving on or approaching a ramp, road surface is always visible
        player = getattr(getattr(self, 'sim', None), 'player', None)
        on_ramp_or_deck = (getattr(player, 'on_bridge', False) if player else False) or (camh > min(z0, z1) - 0.9)
        above = solid or (camh > z_plane - 0.35) or on_ramp_or_deck
        below = (not solid) and (camh < z_plane + 0.25) and not on_ramp_or_deck
        # underside (seen from the road below)
        if below:
            self._quad3([(c0[0] - rx * hd, c0[1] - ry * hd, z0 - th), (c1[0] - rx * hd, c1[1] - ry * hd, z1 - th),
                         (c1[0] + rx * hd, c1[1] + ry * hd, z1 - th), (c0[0] + rx * hd, c0[1] + ry * hd, z0 - th)],
                        lit((96, 94, 90), dist))
        wall_c = lit((176, 174, 168), dist)
        par_c = lit((196, 194, 188), dist)
        osd = ch['open_sides']
        for side in (-1, 1):
            nx, ny = rx * side, ry * side
            outside = (camx - mx) * nx + (camy - my) * ny > 0
            a0 = (c0[0] + rx * hd * side, c0[1] + ry * hd * side)
            a1 = (c1[0] + rx * hd * side, c1[1] + ry * hd * side)
            par = 0.0 if side in osd else 0.9
            if outside:
                lo0 = 0.0 if solid else z0 - th
                lo1 = 0.0 if solid else z1 - th
                self._quad3([(a0[0], a0[1], lo0), (a1[0], a1[1], lo1), (a1[0], a1[1], z1 + par), (a0[0], a0[1], z0 + par)],
                            wall_c)
            elif par > 0.0:
                self._quad3([(a0[0], a0[1], z0), (a1[0], a1[1], z1), (a1[0], a1[1], z1 + 0.9), (a0[0], a0[1], z0 + 0.9)],
                            par_c)
        if above:
            self._quad3([(c0[0] - rx * hd, c0[1] - ry * hd, z0), (c1[0] - rx * hd, c1[1] - ry * hd, z1),
                         (c1[0] + rx * hd, c1[1] + ry * hd, z1), (c0[0] + rx * hd, c0[1] + ry * hd, z0)],
                        lit(ch['side_col'], dist))
            self._quad3([(c0[0] - rx * hw, c0[1] - ry * hw, z0 + 0.02), (c1[0] - rx * hw, c1[1] - ry * hw, z1 + 0.02),
                         (c1[0] + rx * hw, c1[1] + ry * hw, z1 + 0.02), (c0[0] + rx * hw, c0[1] + ry * hw, z0 + 0.02)],
                        lit(ch['road_col'], dist))
            if dist < 260.0:
                for pts, col in ch['marks']:
                    self._quad3(pts, lit(col, dist))
        if ch['median']:
            h2 = 0.42
            for side in (-1, 1):
                nx, ny = rx * side, ry * side
                if (camx - mx) * nx + (camy - my) * ny > 0:
                    a0 = (c0[0] + rx * h2 * side, c0[1] + ry * h2 * side)
                    a1 = (c1[0] + rx * h2 * side, c1[1] + ry * h2 * side)
                    self._quad3([(a0[0], a0[1], z0), (a1[0], a1[1], z1), (a1[0], a1[1], z1 + 0.95), (a0[0], a0[1], z0 + 0.95)],
                                lit((178, 176, 170), dist))
            if above:
                self._quad3([(c0[0] - rx * h2, c0[1] - ry * h2, z0 + 0.95), (c1[0] - rx * h2, c1[1] - ry * h2, z1 + 0.95),
                             (c1[0] + rx * h2, c1[1] + ry * h2, z1 + 0.95), (c0[0] + rx * h2, c0[1] + ry * h2, z0 + 0.95)],
                            lit((204, 202, 196), dist))

    def _draw_pillar(self, p, dist):
        # When camera is elevated, or player is on flyover deck/ramp,
        # ground pillars underneath the bridge are 100% occluded by the solid concrete deck.
        player = getattr(getattr(self, 'sim', None), 'player', None)
        if (self.cam_h > 2.0 or 
            (player and (getattr(player, 'on_bridge', False) or getattr(player, 'z', 0.0) > 1.2))):
            return
        ux, uy = p['ux'], p['uy']
        rx, ry = -uy, ux
        s = p['s']
        lit = self._lit
        col = lit((168, 166, 160), dist)
        col_s = lit((140, 138, 132), dist)
        x, y = p['x'], p['y']
        corners = [(x + ux * s + rx * s, y + uy * s + ry * s), (x + ux * s - rx * s, y + uy * s - ry * s),
                   (x - ux * s - rx * s, y - uy * s - ry * s), (x - ux * s + rx * s, y - uy * s + ry * s)]
        self._box(corners, 0.0, p['h'], [col, col_s, col, col_s], None)
        # cap beam across the deck
        w = 4.2 if s > 0.9 else 3.0
        cap = [(x + ux * 0.9 + rx * w, y + uy * 0.9 + ry * w), (x + ux * 0.9 - rx * w, y + uy * 0.9 - ry * w),
               (x - ux * 0.9 - rx * w, y - uy * 0.9 - ry * w), (x - ux * 0.9 + rx * w, y - uy * 0.9 + ry * w)]
        self._box(cap, p['h'] - 1.1, p['h'], [col_s, col, col_s, col], None)

    # ---------------------------------------------------------- pedestrians
    def _draw_ped(self, p, dist):
        px, pz = self._to_cam(p.x, p.y)
        if pz < 0.7:
            return
        k = self.f / pz
        sx = self.hw + px * k
        if sx < -60 or sx > self.W + 60:
            return
        base = self.hor + self.cam_h * k
        H = p.height * k
        lit = self._lit
        sw = math.sin(p.phase) * (0.28 if p.moving else 0.0) * k
        w = max(1, int(0.16 * k))
        pants = lit(p.pants, dist)
        hip = base - H * 0.50
        pygame.draw.line(self.surf, pants, (sx - 0.1 * k, hip), (sx - 0.1 * k + sw, base), w)
        pygame.draw.line(self.surf, pants, (sx + 0.1 * k, hip), (sx + 0.1 * k - sw, base), w)
        shirt = lit(p.shirt, dist)
        tw = max(2, int(0.46 * k))
        top = base - H * 0.84
        pygame.draw.rect(self.surf, shirt, (int(sx - tw / 2), int(top), tw, int(hip - top)), border_radius=max(1, tw // 4))
        skin = lit((235, 195, 165), dist)
        pygame.draw.line(self.surf, skin, (sx - tw / 2, top + 0.1 * k), (sx - tw / 2 - sw * 0.5, top + 0.55 * k), w)
        pygame.draw.line(self.surf, skin, (sx + tw / 2, top + 0.1 * k), (sx + tw / 2 + sw * 0.5, top + 0.55 * k), w)
        pygame.draw.circle(self.surf, skin, (int(sx), int(base - H * 0.92)), max(2, int(0.13 * k)))

    # ------------------------------------------------------------- vehicles
    @staticmethod
    def _lc(car, xf, xr, yl, yr):
        """Corners of a local box (x forward, y to the right) in world space; edge order = front,left,rear,right."""
        c, s = math.cos(car.heading), math.sin(car.heading)
        X, Y = car.x, car.y

        def W(lx, ly):
            return (X + lx * c - ly * s, Y + lx * s + ly * c)
        return [W(xf, yr), W(xf, yl), W(xr, yl), W(xr, yr)]

    def _fq(self, a, b, u0, u1, z0, z1, col):
        ex, ey = b[0] - a[0], b[1] - a[1]
        ax, ay = a[0] + ex * u0, a[1] + ey * u0
        bx, by = a[0] + ex * u1, a[1] + ey * u1
        self._quad3([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], col)

    def _parts_draw(self, parts):
        camx, camy = self.cam_x, self.cam_y
        keyed = []
        for p in parts:
            cs_ = p[0]
            cx = (cs_[0][0] + cs_[2][0]) * 0.5
            cy = (cs_[0][1] + cs_[2][1]) * 0.5
            keyed.append(((cx - camx) ** 2 + (cy - camy) ** 2 + (0.0 if len(p) < 7 else p[6]), p))
        keyed.sort(key=lambda kp: -kp[0])
        for _, p in keyed:
            self._box(p[0], p[1], p[2], p[3], p[4], p[5])

    def _draw_vehicle(self, car, dist, t, is_player=False):
        saved_zo = self._zo
        self._zo = car.z
        try:
            self._draw_vehicle_inner(car, dist, t, is_player)
        finally:
            self._zo = saved_zo

    def _draw_vehicle_inner(self, car, dist, t, is_player):
        kind = car.body
        spec = VEHICLES[kind]
        lit = self._lit
        col = car.color
        col2 = getattr(car, 'color2', (30, 30, 36))
        braking = getattr(car, 'braking', False)
        night = self.pal['night'] or self.pal_name == 'SUNSET'
        L, Wd = car.length, car.width
        hl, hwd = L * 0.5, Wd * 0.5
        glass = lit((36, 52, 70), dist)
        dark = lit((26, 26, 30), dist)
        body_s = lit(shade(col, 0.82), dist)
        body_t = lit(shade(col, 1.15), dist)
        tail_c = (255, 50, 36) if braking else ((200, 30, 26) if night else (150, 22, 20))
        head_c = (255, 250, 210) if night else (225, 225, 210)
        parts = []
        lc = self._lc
        k = spec['kind']

        # ---- far away: one simple prism ----
        if dist > 150.0 and k not in ('bus',):
            hh = spec['height'] * (0.6 if k in ('bike', 'scooter') else 0.85)
            self._box(lc(car, hl, -hl, -hwd * 0.7, hwd * 0.7), 0.2, hh, [body_s] * 4, body_t)
            return

        def lights(h_ch, zlo=0.5, zhi=0.78):
            def fn(i, a, b):
                if i == 2:
                    for (u0, u1) in ((0.06, 0.26), (0.74, 0.94)):
                        self._fq(a, b, u0, u1, h_ch * zlo, h_ch * zhi, tail_c)
                elif i == 0:
                    for (u0, u1) in ((0.06, 0.26), (0.74, 0.94)):
                        self._fq(a, b, u0, u1, h_ch * zlo, h_ch * zhi, head_c)
            return fn

        if k == 'car':
            h_ch = 0.95 if kind == 'sedan' else 1.10
            rear, front, roof = (0.26, 0.30, 1.45) if kind == 'sedan' else (0.08, 0.28, 1.85)
            # Lower body chassis
            parts.append((lc(car, hl, -hl, -hwd, hwd), 0.22, h_ch, [body_s] * 4, body_t, lights(h_ch), 0.10))
            # Cabin, roof and windshield
            x1, x0 = hl - L * front, -hl + L * rear
            parts.append((lc(car, x1, x0, -(hwd - 0.12), hwd - 0.12), h_ch, roof, [glass] * 4, body_t, None, -0.30))

            # Realistic visible wheels with silver alloy rims and tires
            if dist < 140.0:
                tire_c = lit((36, 38, 42), dist)
                rim_c = lit((215, 222, 230), dist)

                def make_rim_face(sx_sign):
                    out_face = 3 if sx_sign > 0 else 1
                    def rim_decor(face_i, p_a, p_b):
                        if face_i == out_face:
                            self._fq(p_a, p_b, 0.15, 0.85, 0.10, 0.46, rim_c)
                            self._fq(p_a, p_b, 0.35, 0.65, 0.20, 0.36, dark)
                    return rim_decor

                for sx in (-1, 1):
                    rim_fn = make_rim_face(sx)
                    for is_front, xx in ((True, hl * 0.62), (False, -hl * 0.62)):
                        # Protrude slightly outside the chassis for full visibility from all angles
                        y_in = sx * (hwd - 0.10)
                        y_out = sx * (hwd + 0.06)
                        yl = min(y_in, y_out)
                        yr = max(y_in, y_out)

                        # Turning front wheels
                        st_ang = getattr(car, 'steer_angle', 0.0) if is_front else 0.0
                        if abs(st_ang) > 0.02:
                            c_w, s_w = math.cos(car.heading + st_ang), math.sin(car.heading + st_ang)
                            cx_w, cy_w = car.x + math.cos(car.heading) * xx - math.sin(car.heading) * (sx * hwd),                                          car.y + math.sin(car.heading) * xx + math.cos(car.heading) * (sx * hwd)
                            hl_w, hw_w = 0.34, 0.12
                            c_pts = [(cx_w + c_w * hl_w - s_w * hw_w, cy_w + s_w * hl_w + c_w * hw_w),
                                     (cx_w + c_w * hl_w + s_w * hw_w, cy_w + s_w * hl_w - c_w * hw_w),
                                     (cx_w - c_w * hl_w + s_w * hw_w, cy_w - s_w * hl_w - c_w * hw_w),
                                     (cx_w - c_w * hl_w - s_w * hw_w, cy_w - s_w * hl_w + c_w * hw_w)]
                            parts.append((c_pts, 0.0, 0.56, [tire_c] * 4, rim_c, rim_fn, -0.35))
                        else:
                            parts.append((lc(car, xx + 0.35, xx - 0.35, yl, yr),
                                          0.0, 0.56, [tire_c] * 4, rim_c, rim_fn, -0.35))
        elif k == 'truck':
            cargo_c = lit(col, dist)
            cab_c = lit(shade(col, 0.8), dist)
            parts.append((lc(car, hl, -hl, -hwd * 0.95, hwd * 0.95), 0.3, 1.0, [dark] * 4, None, None))
            parts.append((lc(car, hl * 0.18, -hl, -hwd, hwd), 1.0, 2.9, [cargo_c] * 4, lit(shade(col, 1.1), dist), None))
            parts.append((lc(car, hl - 0.05, hl * 0.26, -hwd * 0.92, hwd * 0.92), 1.0, 2.25, [glass, cab_c, cab_c, cab_c],
                          cab_c, lights(1.0, 0.1, 0.5), 0.5))
        elif k == 'bus':
            stripe = lit(col2, dist)
            roof_c = lit((236, 236, 236), dist)
            side_c = lit(col, dist)

            def bus_faces(i, a, b):
                if dist > 130.0:
                    return
                if i in (1, 3):
                    self._fq(a, b, 0.0, 1.0, 0.45, 1.05, stripe)
                    n = max(3, int(L / 1.55))
                    for wi in range(n):
                        u0, u1 = (wi + 0.12) / n, (wi + 0.88) / n
                        self._fq(a, b, 0.04 + u0 * 0.92, 0.04 + u1 * 0.92, 1.4, 2.5, glass)
                    for u in (0.16, 0.82):          # wheel arches
                        self._fq(a, b, u - 0.04, u + 0.04, 0.0, 0.85, dark)
                elif i == 0:
                    self._fq(a, b, 0.07, 0.93, 1.25, 2.65, glass)
                    self._fq(a, b, 0.22, 0.78, 2.72, 2.95, (255, 190, 40))
                    for (u0, u1) in ((0.05, 0.16), (0.84, 0.95)):
                        self._fq(a, b, u0, u1, 0.55, 0.85, head_c)
                elif i == 2:
                    self._fq(a, b, 0.18, 0.82, 1.5, 2.45, glass)
                    for (u0, u1) in ((0.05, 0.16), (0.84, 0.95)):
                        self._fq(a, b, u0, u1, 0.6, 0.95, tail_c)
            parts.append((lc(car, hl, -hl, -hwd, hwd), 0.4, 3.05, [side_c] * 4, roof_c, bus_faces))
        elif k == 'auto':
            lower = lit(col, dist)
            canopy = lit(col2, dist)
            nose = lit(shade(col, 1.05), dist)
            tires = dark

            def cab_faces(i, a, b):
                if i == 0:
                    self._fq(a, b, 0.08, 0.92, 0.2, 0.8, glass)
                elif i == 2:
                    for (u0, u1) in ((0.08, 0.24), (0.76, 0.92)):
                        self._fq(a, b, u0, u1, 0.1, 0.3, tail_c)

            def nose_faces(i, a, b):
                if i == 0:
                    self._fq(a, b, 0.3, 0.7, 0.4, 0.65, head_c)
            for sx in (-1, 1):
                parts.append((lc(car, -hl * 0.65 + 0.32, -hl * 0.65 - 0.32, sx * 0.62 - 0.1, sx * 0.62 + 0.1), 0.0, 0.5, [tires] * 4, None, None))
            parts.append((lc(car, hl * 0.8 + 0.28, hl * 0.8 - 0.28, -0.08, 0.08), 0.0, 0.5, [tires] * 4, None, None))
            parts.append((lc(car, hl * 0.28, -hl, -hwd, hwd), 0.28, 0.95, [lower] * 4, lower, cab_faces))
            parts.append((lc(car, hl * 0.28, -hl, -(hwd - 0.05), hwd - 0.05), 0.95, 1.72, [glass, glass, glass, glass], canopy, cab_faces, 0.3))
            parts.append((lc(car, hl, hl * 0.28, -hwd * 0.5, hwd * 0.5), 0.3, 1.0, [nose] * 4, nose, nose_faces))
        else:   # two-wheelers
            self._draw_two_wheeler(car, dist, parts, k, col, col2, tail_c, head_c)
        self._parts_draw(parts)

    def _draw_two_wheeler(self, car, dist, parts, k, col, col2, tail_c, head_c):
        lit = self._lit
        lc = self._lc
        L = car.length
        hl = L * 0.5
        prim = lit(col, dist)
        prim_t = lit(shade(col, 1.2), dist)
        dark = lit((28, 28, 32), dist)
        gray = lit((70, 72, 78), dist)
        shirt = lit(getattr(car, 'rider', RIDER_COLORS[0]), dist)
        pants = lit((40, 46, 66), dist)
        skin = lit((205, 160, 130), dist)
        helmet = lit((236, 236, 236) if car.id % 2 == 0 else (30, 30, 34), dist)
        tail_light = lambda i, a, b: self._fq(a, b, 0.2, 0.8, 0.2, 0.7, tail_c) if i == 2 else None
        head_light = lambda i, a, b: self._fq(a, b, 0.2, 0.8, 0.2, 0.7, head_c) if i == 0 else None
        if k == 'bike':
            parts.append((lc(car, -0.40, -hl + 0.05, -0.07, 0.07), 0.0, 0.62, [dark] * 4, None, None))
            parts.append((lc(car, hl - 0.05, 0.50, -0.07, 0.07), 0.0, 0.62, [dark] * 4, None, None))
            parts.append((lc(car, 0.55, -0.55, -0.17, 0.17), 0.28, 0.62, [gray] * 4, gray, None))
            parts.append((lc(car, 0.55, 0.0, -0.21, 0.21), 0.62, 0.86, [prim] * 4, prim_t, None))
            parts.append((lc(car, 0.0, -0.85, -0.2, 0.2), 0.62, 0.78, [dark] * 4, dark, None))
            parts.append((lc(car, -0.8, -hl + 0.02, -0.12, 0.12), 0.6, 0.82, [prim] * 4, prim_t, tail_light))
            parts.append((lc(car, hl * 0.95, hl * 0.78, -0.1, 0.1), 0.66, 0.98, [dark] * 4, dark, head_light))
            parts.append((lc(car, 0.5, 0.4, -0.35, 0.35), 0.95, 1.02, [dark] * 4, dark, None))
            for sx in (-1, 1):
                parts.append((lc(car, 0.45, -0.2, sx * 0.17 - 0.07 + 0.07 * sx, sx * 0.17 + 0.07 + 0.07 * sx), 0.38, 0.95, [pants] * 4, pants, None))
            torso_x = (-0.05, -0.45)
            parts.append((lc(car, torso_x[0], torso_x[1], -0.2, 0.2), 0.78, 1.45, [shirt] * 4, shirt, None))
            for sx in (-1, 1):
                parts.append((lc(car, 0.45, -0.1, sx * 0.2 - 0.05, sx * 0.2 + 0.05), 1.05, 1.2, [shirt] * 4, shirt, None))
            parts.append((lc(car, -0.04, -0.32, -0.13, 0.13), 1.45, 1.74, [helmet] * 4, helmet, None))
            if car.id % 5 == 0:
                parts.append((lc(car, -0.45, -0.85, -0.2, 0.2), 0.78, 1.42, [lit((90, 120, 180), dist)] * 4, lit((90, 120, 180), dist), None))
                parts.append((lc(car, -0.5, -0.78, -0.12, 0.12), 1.42, 1.68, [skin] * 4, skin, None))
        else:   # scooter
            parts.append((lc(car, -0.45, -hl + 0.05, -0.06, 0.06), 0.0, 0.48, [dark] * 4, None, None))
            parts.append((lc(car, hl - 0.05, 0.42, -0.06, 0.06), 0.0, 0.48, [dark] * 4, None, None))
            parts.append((lc(car, 0.45, -0.1, -0.21, 0.21), 0.2, 0.32, [prim] * 4, prim_t, None))
            parts.append((lc(car, -0.1, -hl + 0.05, -0.21, 0.21), 0.3, 0.62, [prim] * 4, prim_t, tail_light))
            parts.append((lc(car, 0.66, 0.46, -0.22, 0.22), 0.3, 1.1, [prim] * 4, prim_t, head_light))
            parts.append((lc(car, -0.05, -0.85, -0.19, 0.19), 0.6, 0.74, [dark] * 4, dark, None))
            parts.append((lc(car, 0.62, 0.52, -0.36, 0.36), 1.08, 1.14, [dark] * 4, dark, None))
            for sx in (-1, 1):
                parts.append((lc(car, 0.35, -0.2, sx * 0.17 - 0.06 + 0.06 * sx, sx * 0.17 + 0.06 + 0.06 * sx), 0.5, 0.92, [pants] * 4, pants, None))
            parts.append((lc(car, -0.1, -0.5, -0.2, 0.2), 0.74, 1.40, [shirt] * 4, shirt, None))
            for sx in (-1, 1):
                parts.append((lc(car, 0.5, -0.05, sx * 0.2 - 0.05, sx * 0.2 + 0.05), 1.0, 1.15, [shirt] * 4, shirt, None))
            parts.append((lc(car, -0.1, -0.38, -0.13, 0.13), 1.40, 1.68, [helmet] * 4, helmet, None))
        self._parts_draw(parts)

    def _car_glow(self, car):
        px, pz = self._to_cam(car.x, car.y)
        if pz < 1.0 or pz > 160.0:
            return
        k = self.f / pz
        sx = self.hw + px * k
        sy = self.hor + (self.cam_h - car.z - 0.7) * k
        facing = (self.cam_x - car.x) * math.cos(car.heading) + (self.cam_y - car.y) * math.sin(car.heading)
        col = (160, 150, 100) if facing > 0 else ((180, 30, 24) if not car.braking else (255, 50, 36))
        g = self._glow_sprite(max(4, 0.9 * k), col)
        self.surf.blit(g, (sx - g.get_width() // 2, sy - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)

    def _headlight_cone(self, P):
        h = P.heading
        c, s = math.cos(h), math.sin(h)
        r = (-s, c)
        pts = []
        for (a, b) in ((1.0, -0.8), (45.0, -9.0), (45.0, 9.0), (1.0, 0.8)):
            pts.append((P.x + c * a + r[0] * b, P.y + s * a + r[1] * b))
        cx, cy, cs, sn = self.cam_x, self.cam_y, self.cs, self.sn
        sp = []
        near = Config.Z_NEAR
        gh = self.cam_h - P.z
        for (wx, wy) in pts:
            dx, dy = wx - cx, wy - cy
            z = max(near, dx * cs + dy * sn)
            x = -dx * sn + dy * cs
            k = self.f / z
            sp.append((self.hw + x * k, self.hor + gh * k))
        top = max(0, int(self.hor))
        ov = pygame.Surface((self.W, max(2, self.H - top)), pygame.SRCALPHA)
        shifted = [(p[0], p[1] - top) for p in sp]
        pygame.draw.polygon(ov, (255, 250, 200, 46 if self.pal['night'] else 26), shifted)
        self.surf.blit(ov, (0, top))

    # ------------------------------------------------------------ the scene
    def _push_view(self, surf, f):
        saved = (self.surf, self.W, self.H, self.hw, self.f)
        self.surf = surf
        self.W, self.H = surf.get_size()
        self.hw = self.W * 0.5
        self.f = f
        return saved

    def _pop_view(self, saved):
        self.surf, self.W, self.H, self.hw, self.f = saved

    def render_3d(self, sim, mode):
        """mode: FIRST_PERSON | CHASE | REAR"""
        self.surf = sim.world_surface
        self.W, self.H = sim.width, sim.height
        self.hw = self.W * 0.5
        self.f = Config.FOCAL_LENGTH
        P = sim.player
        spec = P.spec
        hdg = P.heading
        if mode == "FIRST_PERSON":
            self._set_cam(P.x + math.cos(hdg) * spec['eye_fwd'], P.y + math.sin(hdg) * spec['eye_fwd'], hdg,
                          P.z + spec['eye'], Config.HORIZON_Y)
        elif mode == "REAR":
            back = P.length * 0.5 + 0.15
            self._set_cam(P.x - math.cos(hdg) * back, P.y - math.sin(hdg) * back, hdg + math.pi,
                          P.z + max(1.3, spec['eye'] - 0.1), Config.HORIZON_Y + 12)
        else:
            pitch = 0.17 + getattr(P, 'pitch', 0.0) * 0.35
            dback = 9.5 + P.length * 0.6
            self._set_cam(P.x - math.cos(hdg) * dback, P.y - math.sin(hdg) * dback, hdg, P.z + 3.9,
                          Config.HORIZON_Y - self.f * math.tan(pitch))
        R_rad = getattr(Config, 'RENDER_RADIUS', 160.0)
        R_obj = getattr(Config, 'RENDER_OBJ_RADIUS', 140.0)
        self._scene(sim, include_player=(mode == "CHASE"), sky_full=True, R=R_rad, Robj=R_obj,
                    headlights=(mode == "FIRST_PERSON"))

    def render_mirror(self, sim, surf):
        """Rear-view mirror image into `surf` (already mirrored left-right)."""
        P = sim.player
        spec = P.spec
        W, H = surf.get_size()
        saved = self._push_view(surf, (W * 0.5) / 0.80)
        hdg = P.heading
        self._set_cam(P.x + math.cos(hdg) * 0.2, P.y + math.sin(hdg) * 0.2, hdg + math.pi, P.z + spec['eye'] * 0.95,
                      H * 0.52)
        self._scene(sim, include_player=False, sky_full=False, R=125.0, Robj=115.0, headlights=False)
        self._pop_view(saved)
        flipped = pygame.transform.flip(surf, True, False)
        surf.blit(flipped, (0, 0))

    def render_rear_pip(self, sim, surf):
        P = sim.player
        W, H = surf.get_size()
        saved = self._push_view(surf, (W * 0.5) / 1.1)
        hdg = P.heading
        back = P.length * 0.5 + 0.15
        self._set_cam(P.x - math.cos(hdg) * back, P.y - math.sin(hdg) * back, hdg + math.pi, P.z + 1.25, H * 0.45)
        self._scene(sim, include_player=False, sky_full=False, R=110.0, Robj=100.0, headlights=False)
        self._pop_view(saved)

    def render_sensor(self, sim, surf):
        """Render forward-facing vehicle sensor camera into surf for AI autopilot and training frames."""
        P = sim.player
        spec = P.spec
        hdg = P.heading
        W, H = surf.get_size()
        saved = self._push_view(surf, (W * 0.5) / 1.0)
        self._set_cam(P.x + math.cos(hdg) * spec['eye_fwd'],
                      P.y + math.sin(hdg) * spec['eye_fwd'],
                      hdg,
                      P.z + spec['eye'] * 0.95,
                      H * 0.45)
        self._scene(sim, include_player=False, sky_full=False, R=140.0, Robj=120.0,
                    headlights=(self.pal['night'] or self.pal_name == "SUNSET"))
        self._pop_view(saved)

    def _scene(self, sim, include_player, sky_full, R, Robj, headlights):
        self.sim = sim
        city = self.city
        P = sim.player
        t = sim.sim_time
        f = self.f
        surf = self.surf
        self._zo = 0.0
        if sky_full:
            bd = self.backdrop
            off = int((self.yaw * Config.FOCAL_LENGTH) % bd.get_width())
            hor_shift = int(self.hor - Config.HORIZON_Y)
            surf.fill(self.pal['fog'])
            for xx in (-off, bd.get_width() - off):
                if xx < self.W:
                    surf.blit(bd, (xx, hor_shift))
        else:
            gy_ = int(self.hor)
            surf.fill(mix(self.pal['sky_top'], self.pal['sky_hor'], 0.75), (0, 0, self.W, max(0, gy_)))
        gy = max(0, int(self.hor))
        surf.fill(self._lit(COL_GRASS, 300.0), (0, gy, self.W, max(0, self.H - gy)))
        cell = city.ground_hash.cell
        cx, cy, cs, sn = self.cam_x, self.cam_y, self.cs, self.sn
        gx0, gx1 = int((cx - R) // cell), int((cx + R) // cell)
        gy0, gy1 = int((cy - R) // cell), int((cy + R) // cell)
        layers = ([], [], [], [])
        seen = set()
        cells = city.ground_hash.cells
        tanf = self.hw / f + 0.25
        half = cell * 0.75
        for ix in range(gx0, gx1 + 1):
            for iy in range(gy0, gy1 + 1):
                lst = cells.get((ix, iy))
                if not lst:
                    continue
                mx, my = (ix + 0.5) * cell - cx, (iy + 0.5) * cell - cy
                z = mx * cs + my * sn
                if z < -half * 1.5 or z > R:
                    continue
                x = -mx * sn + my * cs
                eff_z = max(0.0, z + half * 1.5)
                if abs(x) > eff_z * tanf + half * 1.6:
                    continue
                for it in lst:
                    k = id(it)
                    if k in seen:
                        continue
                    seen.add(k)
                    layers[it[0]].append(it)
        self._poly_count = len(seen)
        lims = (R * 0.96, R, R * 0.90, R * 0.82)
        for lay in layers:
            for (layer, col, poly, pcx, pcy, rad, maxd) in lay:
                dx, dy = pcx - cx, pcy - cy
                d2 = dx * dx + dy * dy
                z = dx * cs + dy * sn
                if z < -rad:
                    continue
                eff_z = max(0.0, z + rad)
                if abs(-dx * sn + dy * cs) > eff_z * tanf + rad * 1.6 + 45.0:
                    continue
                if d2 > maxd * maxd:
                    continue
                d = math.sqrt(d2)
                if d - rad > lims[layer]:
                    continue
                self._poly_ground(poly, self._lit(col, max(0.0, d - rad * 0.5)))
        if headlights and (self.pal['night'] or self.pal_name == "SUNSET"):
            self._headlight_cone(P)
        # Determine dominant facing traffic signal axis per junction (prevents green+red dual signals when turning)
        sig_best_axis = {}
        for hd in city.signal_heads:
            x, y = hd['x'], hd['y']
            dot_cam = (cx - x) * hd['fx'] + (cy - y) * hd['fy']
            align = hd['fx'] * cs + hd['fy'] * sn
            if dot_cam <= 0.0 or align > -0.25:
                continue
            s_obj = hd['sig']
            if s_obj not in sig_best_axis or align < sig_best_axis[s_obj][0]:
                sig_best_axis[s_obj] = (align, hd['axis'])
        self.sig_best_axis = sig_best_axis

        # ---- objects, painter sorted
        objs = []
        qx0, qy0, qx1, qy1 = cx - Robj, cy - Robj, cx + Robj, cy + Robj

        def visible(ox, oy, rad):
            dx, dy = ox - cx, oy - cy
            z = dx * cs + dy * sn
            if z < -rad or z > Robj + rad:
                return None
            eff_z = max(0.0, z + rad)
            if abs(-dx * sn + dy * cs) > eff_z * tanf + rad + 8.0:
                return None
            return math.hypot(dx, dy)

        for b in city.bld_hash.query_box(qx0, qy0, qx1, qy1):
            d = visible(b.cx, b.cy, b.rad)
            if d is None:
                continue
            key = math.hypot(b.cx - cx, b.cy - cy)
            objs.append((key if not b.wall else key + 400.0, 0, b, max(key, 1.0)))
        for kind, obj in city.prop_hash.query_box(qx0, qy0, qx1, qy1):
            if kind == 'tree':
                d = visible(obj[0], obj[1], 4.0)
                if d is not None and d < Robj * 0.8:
                    objs.append((d, 1, obj, d))
            elif kind == 'lamp':
                # When elevated/on flyover, skip ground lamps that sit under or next to the flyover deck
                if (self.cam_h > 2.0 or P.z > 1.2 or getattr(P, 'on_bridge', False)) and obj[4] < 2.0:
                    if city.deck_z(obj[0], obj[1]) is not None or city.deck_hash.query_box(obj[0] - 8, obj[1] - 8, obj[0] + 8, obj[1] + 8):
                        continue
                d = visible(obj[0], obj[1], 2.0)
                if d is not None and d < Robj * 0.75:
                    objs.append((d, 2, obj, d))
            elif kind == 'sig':
                d = visible(obj['x'], obj['y'], 2.0)
                if d is not None:
                    objs.append((d, 3, obj, d))
            elif kind == 'pillar':
                # When player/camera is on the flyover or elevated, pillars supporting the deck underneath are hidden below road
                if self.cam_h > 2.0 or P.z > 1.2 or getattr(P, 'on_bridge', False):
                    continue
                d = visible(obj['x'], obj['y'], 6.0)
                if d is not None and d < Robj * 0.8:
                    objs.append((d + 50.0, 9, obj, d))
            elif kind == 'sign':
                if (self.cam_h > 2.0 or P.z > 1.2 or getattr(P, 'on_bridge', False)) and obj.get('z', 0.0) < 1.0:
                    if city.deck_z(obj['x'], obj['y']) is not None or city.deck_hash.query_box(obj['x'] - 8, obj['y'] - 8, obj['x'] + 8, obj['y'] + 8):
                        continue
                d = visible(obj['x'], obj['y'], 2.0)
                if d is not None and d < Robj * 0.70:
                    objs.append((d, 11, obj, d))
            elif kind == 'bar':
                if (self.cam_h > 2.0 or P.z > 1.2 or getattr(P, 'on_bridge', False)) and obj[8] == 0:
                    mid_x = (obj[0] + obj[2]) * 0.5
                    mid_y = (obj[1] + obj[3]) * 0.5
                    if city.deck_z(mid_x, mid_y) is not None or city.deck_hash.query_box(mid_x - 6, mid_y - 6, mid_x + 6, mid_y + 6):
                        continue
                d = visible((obj[0] + obj[2]) * 0.5, (obj[1] + obj[3]) * 0.5, 7.0)
                if d is not None and d < Robj * 0.42:
                    objs.append((d, 4, obj, d))
            else:
                d = visible((obj[0] + obj[2]) * 0.5, (obj[1] + obj[3]) * 0.5, 7.0)
                if d is not None and d < Robj * 0.42:
                    objs.append((d, 4, obj, d))
        for ch in city.deck_hash.query_box(qx0, qy0, qx1, qy1):
            d = visible(ch['cx'], ch['cy'], ch['rad'])
            if d is not None:
                # Bridge climbing fix: Deck chunks are road surfaces. Use max extent + depth bias
                # so the road surface is ALWAYS drawn BEFORE vehicles driving on or near it,
                # preventing the deck from ever clipping car wheels or vehicle bodies.
                c0, c1 = ch['c0'], ch['c1']
                d0 = math.hypot(c0[0] - cx, c0[1] - cy)
                d1 = math.hypot(c1[0] - cx, c1[1] - cy)
                deck_depth_key = max(d, d0, d1) + 14.0
                objs.append((deck_depth_key, 8, ch, d))
        for car in sim.traffic.cars:
            dx, dy = car.x - cx, car.y - cy
            cz = dx * cs + dy * sn
            if cz < 1.2:
                continue
            d = visible(car.x, car.y, 6.0)
            if d is not None and d < Robj:
                objs.append((d, 5, car, d))
        for p in sim.peds.peds:
            cz = (p.x - cx) * cs + (p.y - cy) * sn
            if cz < -1.5:
                continue
            if include_player:
                d_to_cam = math.hypot(p.x - cx, p.y - cy)
                if d_to_cam < 1.0:
                    continue
            d = visible(p.x, p.y, 1.0)
            if d is not None and d < Robj * 0.6:
                objs.append((d, 6, p, d))
        # (Airplane scene check removed)
        if include_player:
            d_p = math.hypot(P.x - cx, P.y - cy)
            objs.append((d_p, 7, P, d_p))
        objs.sort(key=lambda o: -o[0])
        for (_, typ, o, d) in objs:
            if typ == 0:
                self._draw_building(o, d, t)
            elif typ == 1:
                self._draw_tree(o, d)
            elif typ == 2:
                self._draw_lamp(o, d)
            elif typ == 3:
                self._draw_signal(o, d, t)
            elif typ == 4:
                self._draw_barrier(o, d)
            elif typ == 5:
                self._draw_vehicle(o, d, t)
                if self.pal['night'] and d < 120.0:
                    self._car_glow(o)
            elif typ == 6:
                self._draw_ped(o, d)
            elif typ == 7:
                o.color = o.paint
                o.color2 = o.paint2
                o.braking = o.brake > 0.1
                self._draw_vehicle(o, d, t, True)
            elif typ == 8:
                self._draw_deck(o, d)
            elif typ == 9:
                self._draw_pillar(o, d)
            elif typ == 10:
                pass
            elif typ == 11:
                self._draw_sign(o, d)

    # ------------------------------------------------------------- top down
    def render_top(self, sim):
        surf = self.surf = sim.world_surface
        P = sim.player
        sc = self.td_scale
        W, H = sim.width, sim.height
        ax, ay = W * 0.5, H * 0.66
        cs, sn = math.cos(P.heading), math.sin(P.heading)
        pal = self.pal
        t = sim.sim_time
        tint = pal['tint']
        city = self.city

        def S(wx, wy):
            dx, dy = wx - P.x, wy - P.y
            fwd = dx * cs + dy * sn
            rgt = -dx * sn + dy * cs
            return (ax + rgt * sc, ay - fwd * sc)

        def T(col):
            return (int(col[0] * tint[0]), int(col[1] * tint[1]), int(col[2] * tint[2]))

        surf.fill(T(COL_GRASS))
        R = max(W, H) / sc * 0.85
        cell = city.ground_hash.cell
        gx0, gx1 = int((P.x - R) // cell), int((P.x + R) // cell)
        gy0, gy1 = int((P.y - R) // cell), int((P.y + R) // cell)
        layers = ([], [], [], [])
        seen = set()
        for ix in range(gx0, gx1 + 1):
            for iy in range(gy0, gy1 + 1):
                for it in city.ground_hash.cells.get((ix, iy), ()):
                    if id(it) in seen:
                        continue
                    seen.add(id(it))
                    layers[it[0]].append(it)
        for lay in layers:
            for (layer, col, poly, pcx, pcy, rad, maxd) in lay:
                if (pcx - P.x) ** 2 + (pcy - P.y) ** 2 > (R + rad) ** 2:
                    continue
                if sc < 3.0 and layer == 2 and rad < 3.0:
                    continue
                pygame.draw.polygon(surf, T(col), [S(*p) for p in poly])
        for hd in city.signal_heads:
            if (hd['x'] - P.x) ** 2 + (hd['y'] - P.y) ** 2 > R * R:
                continue
            st = hd['sig'].state(hd['axis'], t)
            colr = {'R': (255, 40, 30), 'Y': (255, 190, 20), 'G': (30, 235, 90)}[st]
            a = S(hd['stop'][0], hd['stop'][1])
            pygame.draw.circle(surf, colr, (int(a[0]), int(a[1])), max(3, int(0.9 * sc)))
        for b in city.bld_hash.query_box(P.x - R, P.y - R, P.x + R, P.y + R):
            poly = [S(*p) for p in b.poly]
            c = T(shade(b.color, 0.85 if not b.wall else 0.7))
            pygame.draw.polygon(surf, c, poly)
            pygame.draw.polygon(surf, T((40, 40, 46)), poly, 1)
        for kind, obj in city.prop_hash.query_box(P.x - R, P.y - R, P.x + R, P.y + R):
            if kind == 'tree':
                pygame.draw.circle(surf, T((30, 110, 40)), S(obj[0], obj[1]), max(2, int(obj[2] * 0.33 * sc)))
            elif kind == 'bar':
                pygame.draw.polygon(surf, T((190, 190, 184) if obj[6] == 'median' else (150, 154, 162)),
                                    [S(*p) for p in obj[7]])
            elif kind == 'lamp' and obj[4] < 1.0:
                pygame.draw.circle(surf, T((240, 230, 160)), S(obj[0], obj[1]), max(1, int(0.3 * sc)))
        # flyover decks (drawn above the ground, with a darker 'shadow' offset)
        for ch in city.deck_hash.query_box(P.x - R, P.y - R, P.x + R, P.y + R):
            rx, ry = ch['r']
            hd = ch['hd']
            c0, c1 = ch['c0'], ch['c1']
            quad = [(c0[0] - rx * hd, c0[1] - ry * hd), (c1[0] - rx * hd, c1[1] - ry * hd),
                    (c1[0] + rx * hd, c1[1] + ry * hd), (c0[0] + rx * hd, c0[1] + ry * hd)]
            zz = (ch['z0'] + ch['z1']) * 0.5
            k = 0.55 + 0.45 * min(1.0, zz / DECK_H)
            pygame.draw.polygon(surf, T(shade(ch['side_col'], k)), [S(*p) for p in quad])
            hw = ch['hw']
            quad2 = [(c0[0] - rx * hw, c0[1] - ry * hw), (c1[0] - rx * hw, c1[1] - ry * hw),
                     (c1[0] + rx * hw, c1[1] + ry * hw), (c0[0] + rx * hw, c0[1] + ry * hw)]
            pygame.draw.polygon(surf, T(shade(ch['road_col'], 0.7 + 0.6 * min(1.0, zz / DECK_H))), [S(*p) for p in quad2])
            for side in (-1, 1):
                pygame.draw.line(surf, T((200, 200, 195)), S(c0[0] + rx * hd * side, c0[1] + ry * hd * side),
                                 S(c1[0] + rx * hd * side, c1[1] + ry * hd * side), 2)
        dyn = []
        for car in sim.traffic.cars:
            if (car.x - P.x) ** 2 + (car.y - P.y) ** 2 < R * R:
                dyn.append((car.z, 0, car))
        for p in sim.peds.peds:
            dyn.append((0.0, 1, p))
        P.color, P.color2 = P.paint, P.paint2
        P.braking = P.brake > 0.1
        dyn.append((P.z + 0.01, 2, P))
        dyn.sort(key=lambda d: d[0])
        for z, typ, o in dyn:
            if typ == 1:
                pygame.draw.circle(surf, T(o.shirt), S(o.x, o.y), max(2, int(0.45 * sc)))
            else:
                self._td_car(o, S, T, player=(typ == 2))

    def _td_car(self, car, S, T, player=False):
        cor = [S(*p) for p in car.corners()]
        col = car.color
        pygame.draw.polygon(self.surf, T(col), cor)
        pygame.draw.polygon(self.surf, (15, 15, 20), cor, 1)
        c, s = math.cos(car.heading), math.sin(car.heading)
        L, Wd = car.length, car.width
        k = VEHICLES[car.body]['kind']
        if k in ('bike', 'scooter'):
            q = S(car.x - c * L * 0.1, car.y - s * L * 0.1)
            pygame.draw.circle(self.surf, T(getattr(car, 'rider', (210, 60, 50))), (int(q[0]), int(q[1])), max(2, int(0.38 * self.td_scale)))
        elif k == 'bus':
            for u in (-0.3, 0.0, 0.3):
                q = S(car.x + c * L * u, car.y + s * L * u)
                pygame.draw.circle(self.surf, (60, 80, 100), (int(q[0]), int(q[1])), max(2, int(0.5 * self.td_scale)))
        else:
            ws = [S(car.x + c * (L * 0.12) - s * y, car.y + s * (L * 0.12) + c * y) for y in (-Wd * 0.4, Wd * 0.4)] + \
                 [S(car.x + c * (L * 0.30) - s * y, car.y + s * (L * 0.30) + c * y) for y in (Wd * 0.4, -Wd * 0.4)]
            pygame.draw.polygon(self.surf, (40, 60, 80), ws)
        if getattr(car, 'braking', False):
            for y in (-Wd * 0.38, Wd * 0.38):
                q = S(car.x - c * L * 0.5 - s * y, car.y - s * L * 0.5 + c * y)
                pygame.draw.circle(self.surf, (255, 40, 30), (int(q[0]), int(q[1])), 2)
        if player:
            q = S(car.x, car.y)
            pygame.draw.circle(self.surf, (255, 255, 255), (int(q[0]), int(q[1])), max(3, int(0.3 * self.td_scale)), 1)
# =============================================================================
# AUTOPILOT / NEURAL NETWORK INFERENCE  (unchanged interface)
# =============================================================================
def predict_control(model, obs_rgb, device):
    """
    Returns (steering, throttle) in [-1, 1] from a PyTorch CNN.
    obs_rgb: uint8 array (OBS_HEIGHT, OBS_WIDTH, 3).
    """
    global torch, TORCH_AVAILABLE
    if torch is None:
        try:
            import torch
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
            return 0.0, 0.0
    if model is None:
        return 0.0, 0.0
    img = obs_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        out_np = output.detach().cpu().numpy().squeeze()
    if np.ndim(out_np) == 0 or (isinstance(out_np, np.ndarray) and out_np.size == 1):
        steer = float(out_np)
        throttle = 0.45
    else:
        steer = float(out_np[0])
        throttle = float(out_np[1])
    return clamp(steer, -1.0, 1.0), clamp(throttle, -1.0, 1.0)




# =============================================================================
# SIMULATOR
# =============================================================================
MAP_COLORS = {'H': (255, 168, 60), 'A': (255, 120, 160), 'S': (232, 232, 236), 'L': (196, 164, 120), 'O': (120, 200, 255)}
VIEW_ORDER = ["FIRST_PERSON", "CHASE", "TOP_DOWN"]


class Simulator:
    def __init__(self, model_path=None, seed=None, headless=False, target_fps=60,
                 spawn=0, traffic=None, vehicle='sedan', rl_model=None):
        self.headless = headless
        self.target_fps = target_fps
        if seed is None:
            seed = random.randrange(1000, 999999)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed % (2 ** 31))
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.width, self.height = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        if not headless:
            pygame.display.set_caption("City Driving Simulator")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.world_surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        self.font_large = pygame.font.Font(None, 28)
        self.font_banner = pygame.font.Font(None, 50)
        self.font_speed = pygame.font.Font(None, 60)
        self.mirror_surf = pygame.Surface((330, 98))
        self.pip_surf = pygame.Surface((340, 190))
        self._mirror_tick = 0

        self.joystick = None
        if pygame.joystick.get_count() > 0:
            try:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"[INPUT] Gamepad: {self.joystick.get_name()}")
            except Exception as e:
                print(f"[INPUT] Gamepad notice: {e}")

        self.vehicle = vehicle if vehicle in PLAYER_VEHICLES else 'sedan'
        self.player = Car(self.vehicle)
        if traffic is not None:
            Config.TRAFFIC_COUNT = traffic
        self.spawn_idx = spawn
        self.view_mode = Config.DEFAULT_VIEW
        self.prev_view = self.view_mode
        self.show_map = False
        self.ai_model = None
        self.ai_device = None
        self.ai_mode = False
        self.is_recording = False
        self.recorded_frame_count = 0
        self.frame_index = 0
        self.crash_count = 0
        self.total_distance = 0.0
        self.run_distance = 0.0
        self.best_run = 0.0
        self.crashed = False
        self.crash_reason = ""
        self.crash_time = 0.0
        self.shield = 0.0
        self.sim_time = 0.0
        self.sim_start = time.time()
        self.step_count = 0
        self.warn_text = ""
        self.warn_t = 0.0
        self.palette_name = "DAY"
        self._obs = np.zeros((Config.OBS_HEIGHT, Config.OBS_WIDTH, 3), np.uint8)
        self.slider_speed = Slider(300, 42, 190, 14, Config.SPEED_SLIDER_MIN * 3.6, Config.SPEED_SLIDER_MAX * 3.6,
                                   Config.MAX_SPEED * 3.6, "Max Speed", "{:.0f} km/h")
        self.slider_steer = Slider(300, 90, 190, 14, 0.3, 1.6, Config.STEER_SENSITIVITY, "Turn Sensitivity", "{:.2f}x")
        self.dest_idx = 0
        self.unparking = False
        self.unparking_dest_idx = 0
        self.unparking_curb_pt = None
        self.mission_completed = False
        self.mission_complete_t = -999.0
        self._completed_dest_idx = -1
        self.nav_route = None
        self.nav_full_pts = []
        self.nav_route_pts = []
        self.nav_prog_idx = 0
        self.nav_dest_idx = -1
        self.nav_dist = 0.0
        self.nav_eta = 0.0
        self._last_nav_t = 0.0
        self._landmark_rects = []
        self._landmark_pins = []
        self.rl_model_path = rl_model
        self.auto_drive_active = False
        if rl_model is None and os.path.exists("rl_policy.pt"):
            rl_model = "rl_policy.pt"
        self.auto_drive_controller = AutoDriveController(model_path=rl_model)
        self.is_paused = False
        self.dest_picker_open = False
        self.btn_auto_drive = pygame.Rect(16, 126, 120, 24)
        self.btn_pause = pygame.Rect(142, 126, 100, 24)
        self._picker_rects = []
        self._build_world(seed)
        if model_path:
            self._load_pytorch_model(model_path)
        self._respawn()

    # ---------------------------------------------------------------- world
    def _build_world(self, seed):
        print(f"[CITY] Generating random city (seed {seed}) ...")
        t0 = time.time()
        self.seed = seed
        random.seed(seed)
        self.city = CityMap(seed=seed)
        self.router = AStarRouter(self.city)
        n, e, l, c, b, f = self.city.stats
        print(f"[CITY] done in {time.time() - t0:.1f}s  -  {e} roads, {l} lanes, {c} crosswalks, {b} buildings, {f} flyovers")
        print(f"[CITY] Reproduce this exact city with:  python driving_sim.py --seed {seed}")
        self.renderer = Renderer(self.city, self.width, self.height)
        self.renderer.set_palette(self.palette_name)
        self.traffic = TrafficManager(self.city, seed)
        self.peds = PedestrianManager(self.city, seed)
        self._build_map_surface()
        self.spawn_idx %= len(self.city.spawns)

    def new_city(self):
        self._build_world(random.randrange(1000, 999999))
        self.spawn_idx = 0
        self._respawn()
        self._warn(f"New random city!  seed {self.seed}")

    def _load_pytorch_model(self, model_path):
        global torch, TORCH_AVAILABLE
        if torch is None:
            try:
                import torch
                TORCH_AVAILABLE = True
            except ImportError:
                TORCH_AVAILABLE = False
        if not TORCH_AVAILABLE:
            print("[AUTOPILOT ERROR] PyTorch is not installed. Run: pip install torch")
            return
        if not os.path.exists(model_path):
            print(f"[AUTOPILOT ERROR] Model file not found: {model_path}")
            return
        try:
            dev = "cpu"
            if torch.cuda.is_available():
                try:
                    torch.zeros(1, device="cuda")
                    dev = "cuda"
                except Exception:
                    dev = "cpu"
            self.ai_device = torch.device(dev)
            try:
                self.ai_model = torch.jit.load(model_path, map_location=self.ai_device)
            except Exception:
                try:
                    self.ai_model = torch.load(model_path, map_location=self.ai_device, weights_only=False)
                except TypeError:
                    self.ai_model = torch.load(model_path, map_location=self.ai_device)
            self.ai_model.to(self.ai_device)
            self.ai_model.eval()
            self.ai_mode = True
            print("[AUTOPILOT] Model loaded. Press P to toggle.")
        except Exception as e:
            print(f"[AUTOPILOT ERROR] {e}")
            self.ai_model = None

    # -------------------------------------------------------------- respawn
    def _respawn(self):
        sp = self.city.spawns[self.spawn_idx % len(self.city.spawns)]
        lane, s = sp['lane'], sp['s']
        x, y, h = lane.pose(s)
        P = self.player
        P.set_vehicle(self.vehicle)
        z = lane.zat(s) if lane.elev is not None else 0.0
        P.reset(x, y, h, speed=10.0 if lane.edge.rt.rank >= 2 else 0.0)
        P.z = z
        P.on_bridge = z > 0.3
        P.lvl = 1 if z > 3.2 else 0
        self.traffic.reset((x, y), None, lane, s)
        self.traffic.clear_around(x, y, 24.0, ahead_ok=True)
        self.peds.reset()
        self.crashed = False
        self.crash_reason = ""
        self.shield = Config.RESPAWN_SHIELD_SEC
        self.run_distance = 0.0
        self.spawn_label = sp['label']
        self.update_navigation(force=True)

    def _recompute_route(self, P, dest):
        if not hasattr(self, 'router') or self.router is None:
            self.router = AStarRouter(self.city)

        dest_pos = (dest['x'], dest['y'])
        approach_pts = []
        if dest.get('entry_pt'):
            approach_pts.append(dest['entry_pt'])
        if dest.get('finish_pt'):
            approach_pts.append(dest['finish_pt'])

        dist_direct = math.hypot(dest['x'] - P.x, dest['y'] - P.y)
        if dist_direct < 25.0:
            self.nav_route = []
            pts = [(P.x, P.y)]
            if approach_pts:
                pts.extend(approach_pts)
            else:
                pts.append(dest_pos)
            self.nav_full_pts = pts
            self.nav_route_pts = list(pts)
            self.nav_prog_idx = 0
            self.nav_dest_idx = self.dest_idx
            self.nav_dist = sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))
            self.nav_eta = self.nav_dist / 14.0
            return True

        route, end_s, curb_pt = self.router.plan_optimal_route(P.x, P.y, P.heading, P.z, dest)
        if route is None:
            return False

        self.nav_route = route
        clean_pts = self.router.build_polyline(P.x, P.y, route, end_s, curb_pt, dest_pos, approach_pts)

        self.nav_full_pts = clean_pts
        self.nav_route_pts = list(clean_pts)
        self.nav_prog_idx = 0
        self.nav_dest_idx = self.dest_idx

        tot_len = 0.0
        for k in range(len(clean_pts) - 1):
            tot_len += math.hypot(clean_pts[k + 1][0] - clean_pts[k][0],
                                  clean_pts[k + 1][1] - clean_pts[k][1])
        self.nav_dist = tot_len
        self.nav_eta = tot_len / 14.0
        return True

    def select_destination(self, idx):
        if getattr(self.city, 'destinations', None):
            prev_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            prev_dest = self.city.destinations[prev_idx]
            new_idx = idx % len(self.city.destinations)
            dest = self.city.destinations[new_idx]

            finish_pt = prev_dest.get('finish_pt', (prev_dest['x'], prev_dest['y']))
            d_finish_prev = math.hypot(self.player.x - finish_pt[0], self.player.y - finish_pt[1])

            # Check if vehicle is currently parked inside landmark bay or near finish
            if getattr(self, 'mission_completed', False) or d_finish_prev <= 6.0:
                self.unparking = True
                self.unparking_dest_idx = new_idx
                self.mission_completed = False
                self.auto_drive_active = True
                self.dest_idx = new_idx

                curb_x, curb_y = prev_dest.get('curb_pt', finish_pt)
                app_lane = prev_dest.get('lane')
                out_x, out_y = prev_dest.get('out_vec', (0.0, 1.0))

                if app_lane is not None:
                    s_curb = 0.0
                    min_d = 1e9
                    for k in range(0, int(app_lane.length) + 1):
                        px, py, _ = app_lane.pose(float(k))
                        d = math.hypot(px - curb_x, py - curb_y)
                        if d < min_d:
                            min_d, s_curb = d, float(k)
                    s_final = max(2.5, min(app_lane.length - 2.5, s_curb - 6.0))
                    final_x, final_y, final_hdg = app_lane.pose(s_final)
                    lane_fx, lane_fy = math.cos(final_hdg), math.sin(final_hdg)
                else:
                    fx, fy = prev_dest.get('f_vec', (1.0, 0.0))
                    final_x, final_y = curb_x - out_x * 4.0, curb_y - out_y * 4.0
                    final_hdg = math.atan2(fy, fx)
                    lane_fx, lane_fy = fx, fy

                P0 = (self.player.x, self.player.y)
                P3 = (final_x, final_y)
                d_tot = math.hypot(P3[0] - P0[0], P3[1] - P0[1])
                d1 = max(5.0, min(10.0, d_tot * 0.4))
                d2 = max(5.0, min(10.0, d_tot * 0.4))
                P1 = (P0[0] - out_x * d1, P0[1] - out_y * d1)
                P2 = (P3[0] + lane_fx * d2, P3[1] + lane_fy * d2)

                raw_curve = bezier3(P0, P1, P2, P3, 40)
                unpark_traj = []
                for k in range(len(raw_curve)):
                    pt = raw_curve[k]
                    if k < len(raw_curve) - 1:
                        nxt = raw_curve[k + 1]
                        tangent = math.atan2(nxt[1] - pt[1], nxt[0] - pt[0])
                        hdg = (tangent + math.pi) % (2.0 * math.pi)
                    else:
                        hdg = final_hdg
                    unpark_traj.append((pt[0], pt[1], hdg))

                self.unpark_traj = unpark_traj
                self.unpark_prog = 0
                self.unpark_final_pose = (final_x, final_y, final_hdg)
                self._warn(f"REVERSING OUT OF {prev_dest['name'].upper()} BAY -> DESTINATION: {dest['name'].upper()}")
            else:
                self.dest_idx = new_idx
                self.mission_completed = False
                self.update_navigation(force=True)
                self._warn(f"GPS Target [{self.dest_idx + 1}]: {dest['name']} (A* routed)")

    def _check_rear_hazards(self) -> Tuple[bool, str]:
        """
        Monitors rear path, blind spots, upcoming trajectory points, and approaching
        cross-traffic vehicles/pedestrians while reversing.
        Returns (is_blocked, description).
        """
        P = self.player
        cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
        rear_x = P.x - cos_h * (P.length * 0.5 + 0.4)
        rear_y = P.y - sin_h * (P.length * 0.5 + 0.4)

        # 1. Immediate rear corridor obstacle check (strictly behind rear bumper in vehicle lane width)
        if hasattr(self, 'traffic') and hasattr(self.traffic, 'cars'):
            for car in self.traffic.cars:
                dx = car.x - rear_x
                dy = car.y - rear_y
                fwd = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                # Strictly behind bumper (-3.5m <= fwd <= -0.15m) and within vehicle width (|lat| <= 1.35m)
                if -3.5 <= fwd <= -0.15 and abs(lat) <= 1.35:
                    d_rear = abs(fwd)
                    return True, f"CAR IN REAR PATH ({d_rear:.1f}m)"

        # 2. Pedestrians behind car in our path
        if hasattr(self, 'peds') and hasattr(self.peds, 'peds'):
            for ped in self.peds.peds:
                dx = ped.x - rear_x
                dy = ped.y - rear_y
                fwd = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                if -3.2 <= fwd <= -0.10 and abs(lat) <= 1.25:
                    d_ped = abs(fwd)
                    return True, f"PEDESTRIAN BEHIND CAR ({d_ped:.1f}m)"

        # 3. Path collision check with upcoming unpark trajectory waypoints
        traj = getattr(self, 'unpark_traj', None)
        prog = getattr(self, 'unpark_prog', 0)
        if traj and prog < len(traj):
            look_pts = traj[prog : min(len(traj), prog + 8)]
            if hasattr(self, 'traffic') and hasattr(self.traffic, 'cars'):
                for car in self.traffic.cars:
                    for wp in look_pts:
                        if math.hypot(car.x - wp[0], car.y - wp[1]) < 2.2:
                            return True, "TRAFFIC BLOCKING REVERSE PATH"

        # 4. Approaching street cross-traffic check (ONLY when vehicle is backing out into street: prog >= len(traj) - 16)
        final_pose = getattr(self, 'unpark_final_pose', None)
        if final_pose and traj and prog >= max(0, len(traj) - 16) and hasattr(self, 'traffic') and hasattr(self.traffic, 'cars'):
            final_x, final_y, final_hdg = final_pose
            for car in self.traffic.cars:
                dx = final_x - car.x
                dy = final_y - car.y
                d_merge = math.hypot(dx, dy)
                if d_merge < 20.0:
                    car_v = getattr(car, 'v', 0.0)
                    car_hdg = getattr(car, 'heading', 0.0)
                    dot_fwd = dx * math.cos(car_hdg) + dy * math.sin(car_hdg)
                    hdg_align = math.cos(car_hdg - final_hdg)
                    if dot_fwd > 0.0 and hdg_align > 0.5 and car_v > 0.8:
                        ttc = d_merge / max(0.5, car_v)
                        if ttc < 4.0:
                            return True, f"CROSS-TRAFFIC ON ROAD ({d_merge:.0f}m)"

        return False, ""

    def _unpark_step(self, dt):
        """
        Autonomous smooth reverse unparking maneuver:
        Reverses smoothly out along the Bézier unpark curve directly into the street lane,
        with cross-traffic perception, pedestrian yielding, and continuous heading rotation.
        """
        P = self.player
        traj = getattr(self, 'unpark_traj', None)
        final_pose = getattr(self, 'unpark_final_pose', None)

        if not traj or not final_pose or self.unpark_prog >= len(traj) - 1:
            if final_pose:
                P.x, P.y, P.heading = final_pose
            P.v = 0.0
            P.brake = 1.0
            P.throttle = 0.0
            P.steer_angle = 0.0
            P.steer_in = 0.0
            P.allow_reverse = False
            self.unparking = False
            self.unparking_blocked = False
            self.dest_idx = getattr(self, 'unparking_dest_idx', self.dest_idx)
            self.update_navigation(force=True)
            d_name = self.city.destinations[self.dest_idx]['name']
            self._warn(f"★ SAFELY UNPARKED IN LANE! PROCEEDING TO {d_name.upper()} ★")
            return

        # Rear cross-traffic and obstacle perception check:
        is_blocked, block_reason = self._check_rear_hazards()
        if is_blocked:
            self.unparking_blocked = True
            self.unparking_blocked_reason = block_reason
            self.unparking_yield_timer = 0.35
            self.unpark_blocked_timer = getattr(self, 'unpark_blocked_timer', 0.0) + dt
            P.v = 0.0
            P.brake = 1.0
            P.throttle = 0.0
            P.allow_reverse = True

            # Watchdog timeout: if traffic has queued behind reverse path for > 2.0s, break deadlock!
            # Abort reverse and proceed directly forward onto route:
            if self.unpark_blocked_timer > 2.0:
                self.unparking = False
                self.unparking_blocked = False
                self.unpark_blocked_timer = 0.0
                P.brake = 0.0
                P.throttle = 0.3
                self.dest_idx = getattr(self, 'unparking_dest_idx', self.dest_idx)
                self.update_navigation(force=True)
                d_name = self.city.destinations[self.dest_idx]['name']
                self._warn(f"TRAFFIC BEHIND: ABORTING REVERSE -> DRIVING FORWARD TO {d_name.upper()}")
            return
        else:
            self.unpark_blocked_timer = 0.0
            if getattr(self, 'unparking_yield_timer', 0.0) > 0.0:
                self.unparking_yield_timer -= dt
                P.v = 0.0
                P.brake = 1.0
                P.throttle = 0.0
                P.allow_reverse = True
                return
            else:
                self.unparking_blocked = False
                self.unparking_blocked_reason = ""

        target_pt = traj[self.unpark_prog]
        dist_to_wp = math.hypot(target_pt[0] - P.x, target_pt[1] - P.y)
        if dist_to_wp < 0.9 and self.unpark_prog < len(traj) - 1:
            self.unpark_prog += 1
            target_pt = traj[self.unpark_prog]

        rem_steps = len(traj) - 1 - self.unpark_prog
        rev_speed = max(0.8, rem_steps * 0.4) if rem_steps <= 6 else min(3.0, (self.unpark_prog + 3) * 0.5)

        P.allow_reverse = True
        P.v = -rev_speed

        dx = target_pt[0] - P.x
        dy = target_pt[1] - P.y
        d_step = rev_speed * dt
        step_len = math.hypot(dx, dy)
        if step_len > 1e-4:
            P.x += (dx / step_len) * min(d_step, step_len)
            P.y += (dy / step_len) * min(d_step, step_len)

        target_hdg = target_pt[2]
        hdg_err = (target_hdg - P.heading + math.pi) % (2.0 * math.pi) - math.pi
        P.heading = (P.heading + hdg_err * min(1.0, 16.0 * dt)) % (2.0 * math.pi)
        steer = float(np.clip(-hdg_err * 0.8, -0.65, 0.65))
        P.steer_angle = steer * 0.5
        P.steer_in = steer

    def update_navigation(self, force=False):
        if not getattr(self.city, 'destinations', None):
            self.nav_route = None
            self.nav_full_pts = []
            self.nav_route_pts = []
            self.nav_prog_idx = 0
            self.nav_dest_idx = -1
            self.nav_dist = 0.0
            self.nav_eta = 0.0
            return

        P = self.player
        d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
        dest = self.city.destinations[d_idx]

        # 1. Force recalculation or destination changed
        need_recompute = (force or 
                          getattr(self, 'nav_dest_idx', -1) != d_idx or 
                          not getattr(self, 'nav_full_pts', None) or 
                          len(self.nav_full_pts) < 2)

        if need_recompute:
            self._recompute_route(P, dest)
            return

        # 2. Rate limit progress check to ~10 Hz (every 0.1s simulation time)
        now = self.sim_time
        if hasattr(self, '_last_nav_t') and (now - self._last_nav_t) < 0.1:
            return
        self._last_nav_t = now

        pts = self.nav_full_pts
        px, py = P.x, P.y
        OFF_ROUTE_THRESHOLD = 30.0  # metres

        # 3. Check distance from player to active route segments ahead
        start_seg = max(0, getattr(self, 'nav_prog_idx', 0) - 1)
        window_end = min(len(pts) - 1, start_seg + 35)

        best_d = 1e9
        best_idx = start_seg

        for i in range(start_seg, window_end):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 < 1e-6:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
                proj_x = ax + t * dx
                proj_y = ay + t * dy
                d = math.hypot(px - proj_x, py - proj_y)
            if d < best_d:
                best_d = d
                best_idx = i

        if best_d > OFF_ROUTE_THRESHOLD and window_end < len(pts) - 1:
            for i in range(window_end, len(pts) - 1):
                ax, ay = pts[i]
                bx, by = pts[i + 1]
                dx, dy = bx - ax, by - ay
                l2 = dx * dx + dy * dy
                if l2 < 1e-6:
                    d = math.hypot(px - ax, py - ay)
                else:
                    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
                    proj_x = ax + t * dx
                    proj_y = ay + t * dy
                    d = math.hypot(px - proj_x, py - proj_y)
                if d < best_d:
                    best_d = d
                    best_idx = i

        # 4. If vehicle is following the first given route:
        # DO NOT change the route! Keep path stable and advance smoothly.
        if best_d <= OFF_ROUTE_THRESHOLD:
            self.nav_prog_idx = max(getattr(self, 'nav_prog_idx', 0), best_idx)
            remaining_pts = [(px, py)]
            if self.nav_prog_idx + 1 < len(pts):
                remaining_pts.extend(pts[self.nav_prog_idx + 1:])
            else:
                remaining_pts.append(pts[-1])
            self.nav_route_pts = remaining_pts

            rem_dist = 0.0
            for k in range(len(remaining_pts) - 1):
                rem_dist += math.hypot(remaining_pts[k + 1][0] - remaining_pts[k][0],
                                       remaining_pts[k + 1][1] - remaining_pts[k][1])
            self.nav_dist = rem_dist
            self.nav_eta = rem_dist / 14.0
        else:
            # 5. Vehicle DID NOT follow the first given route!
            # Only NOW change the path by recomputing A* from the new location.
            if self._recompute_route(P, dest):
                self._warn(f"Rerouting to {dest['name']}...")

    def _handle_map_click(self, pos):
        # 1. Check sidebar landmark options
        for (rect, idx) in getattr(self, '_landmark_rects', []):
            if rect.collidepoint(pos):
                self.select_destination(idx)
                return
        # 2. Check landmark pins on map
        for (q, idx) in getattr(self, '_landmark_pins', []):
            if math.hypot(pos[0] - q[0], pos[1] - q[1]) < 22:
                self.select_destination(idx)
                return

    def set_time_of_day(self, name):
        name = name.upper()
        if name not in PALETTES:
            name = "DAY"
        self.palette_name = name
        self.renderer.set_palette(name)
        print(f"[TIME] {name}")

    # ------------------------------------------------------------ map image
    def _build_map_surface(self):
        c = self.city
        x0, y0, x1, y1 = c.bounds
        self.map_origin = (x0, y0)
        w, h = int(x1 - x0) + 2, int(y1 - y0) + 2
        surf = pygame.Surface((w, h))
        surf.fill((42, 92, 44))
        for b in c.buildings:
            if b.wall:
                continue
            bx, by, bw, bh = b.x0 - x0, b.y0 - y0, b.x1 - b.x0, b.y1 - b.y0
            if getattr(b, 'landmark', None):
                l_col = b.landmark.get('color', (240, 200, 80))
                pygame.draw.rect(surf, l_col, (bx, by, bw, bh))
                pygame.draw.rect(surf, (20, 24, 30), (bx, by, bw, bh), 2)
            else:
                pygame.draw.rect(surf, (118, 112, 106), (bx, by, bw, bh))
        seen = set()
        items = []
        for lst in c.ground_hash.cells.values():
            for it in lst:
                if id(it) not in seen:
                    seen.add(id(it))
                    items.append(it)
        items.sort(key=lambda it: it[0])
        colmap = {rt.road_col: MAP_COLORS[k] for k, rt in RT.items()}
        for (layer, col, poly, *_r) in items:
            pts = [(p[0] - x0, p[1] - y0) for p in poly]
            if layer == 0:
                pygame.draw.polygon(surf, (150, 148, 142), pts)
            elif layer == 1:
                pygame.draw.polygon(surf, colmap.get(col, (226, 226, 230)), pts)
            elif layer == 2 and col == COL_MEDIAN:
                pygame.draw.polygon(surf, (70, 150, 70), pts)
        for ch in c.decks:
            rx, ry = ch['r']
            hw = ch['hw']
            c0, c1 = ch['c0'], ch['c1']
            q = [(c0[0] - rx * hw - x0, c0[1] - ry * hw - y0), (c1[0] - rx * hw - x0, c1[1] - ry * hw - y0),
                 (c1[0] + rx * hw - x0, c1[1] + ry * hw - y0), (c0[0] + rx * hw - x0, c0[1] + ry * hw - y0)]
            pygame.draw.polygon(surf, (176, 96, 255), q)
        self.map_surf = surf

    def mp(self, x, y):
        return x - self.map_origin[0], y - self.map_origin[1]

    # ------------------------------------------------------ observation / io
    def get_observation(self):
        if getattr(self, '_obs_frame', -1) == self.frame_index and hasattr(self, '_cached_obs'):
            return self._cached_obs
        if not hasattr(self, 'sensor_surf_hires'):
            self.sensor_surf_hires = pygame.Surface((400, 132))
        self.renderer.render_sensor(self, self.sensor_surf_hires)
        scaled = pygame.transform.smoothscale(self.sensor_surf_hires, (Config.OBS_WIDTH, Config.OBS_HEIGHT))
        self._sensor_display_surf = scaled
        arr = pygame.surfarray.array3d(scaled)
        self._cached_obs = np.transpose(arr, (1, 0, 2))
        self._obs_frame = self.frame_index
        return self._cached_obs

    def _save_recorded_frame(self, obs):
        os.makedirs(Config.IMAGE_DIR, exist_ok=True)
        path = os.path.join(Config.IMAGE_DIR, f"frame_{self.recorded_frame_count:06d}.jpg")
        if CV2_AVAILABLE:
            cv2.imwrite(path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        else:
            pygame.image.save(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), path)
        write_header = not os.path.exists(Config.LOG_FILE)
        with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
            if write_header:
                f.write("image_path,steering,throttle,speed,timestamp\n")
            f.write(f"{path},{self.player.norm_steer:.4f},{self.player.norm_throttle:.4f},"
                    f"{self.player.v:.3f},{time.time():.4f}\n")
        self.recorded_frame_count += 1

    # ------------------------------------------------------------ collisions
    def _check_collisions(self):
        if self.shield > 0.0:
            return ""
        P = self.player
        cor = P.corners()
        r = self.city.static_collision(cor, P.lvl)
        if r:
            return r
        for c in self.traffic.cars:
            if abs(c.x - P.x) < 8.0 and abs(c.y - P.y) < 8.0 and abs(c.z - P.z) < 2.6:
                if sat_polygon_collision(cor, c.corners()):
                    if math.cos(c.heading - P.heading) < -0.3:
                        return "HEAD-ON COLLISION WITH ONCOMING VEHICLE (INCOMING LINE)!"
                    return "HIT ANOTHER VEHICLE!"
        if abs(P.v) >= 0.5 and P.z < 2.0:
            for p in self.peds.peds:
                if abs(p.x - P.x) < 4.0 and abs(p.y - P.y) < 4.0:
                    if circle_obb_collision(p.x, p.y, 0.36, P.x, P.y, P.heading, P.length, P.width):
                        return "HIT A PEDESTRIAN!"
        return ""

    def _crash(self, reason):
        self.crashed = True
        self.crash_reason = reason
        self.crash_time = self.sim_time
        self.crash_count += 1
        self.player.cruise_on = False

        # Online AI Crash Reflection & Adaptation
        if hasattr(self, 'auto_drive_controller') and self.auto_drive_controller is not None:
            report = self.auto_drive_controller.on_crash(self, reason)
            self.latest_crash_report = report
            self.auto_drive_controller.live_telemetry['mistakes_learned'] = self.auto_drive_controller.crash_learner.total_mistakes_learned
            print(f"[CRASH #{self.crash_count}] {reason} | AI Reflection: {report.diagnosis}")
            print(f"  -> Solution Applied: {report.remediation}")
        else:
            print(f"[CRASH #{self.crash_count}] {reason} (run distance {self.run_distance:.0f} m)")

    def toggle_auto_drive(self):
        """Toggles the Auto Drive autonomous mode reliably with arrival and crash recovery."""
        self.auto_drive_active = not self.auto_drive_active
        self._auto_drive_engage_t = self.sim_time
        if not self.auto_drive_active:
            self.unparking = False
            self.unparking_blocked = False
            self.player.brake = 0.0
        if self.auto_drive_active:
            self.ai_mode = False  # Mutual exclusion with CNN autopilot
            self.player.cruise_on = False
            self.dest_picker_open = False
            self.crashed = False  # Clear crash state so player can drive immediately
            self.crash_reason = ""
            
            # If parked at previous destination, advance to next destination!
            d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            dest = self.city.destinations[d_idx]
            finish_pt = dest.get('finish_pt', (dest['x'], dest['y']))
            d_finish = math.hypot(self.player.x - finish_pt[0], self.player.y - finish_pt[1])
            if getattr(self, 'mission_completed', False) or d_finish <= 3.0:
                next_idx = (d_idx + 1) % len(self.city.destinations)
                self.mission_completed = False
                self.select_destination(next_idx)
            else:
                self.mission_completed = False
                self.update_navigation(force=True)

            d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            d_name = self.city.destinations[d_idx]['name']
            has_rl = self.auto_drive_controller.has_model
            mode_str = "RL Policy" if has_rl else "Rule-Based + Safety"
            self._warn(f"AUTO DRIVE ACTIVE ({mode_str}) -> {d_name}")
        else:
            self.unparking = False
            self._warn("AUTO DRIVE OFF - Manual control restored")

    def toggle_pause(self):
        """Toggles simulation pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self._warn("SIMULATION PAUSED [TAB / RESUME]")
        else:
            self._warn("SIMULATION RESUMED")

    def get_rl_observation(self):
        """Extracts normalized 28-dimensional state vector for RL policy & Auto Drive."""
        P = self.player
        speed = abs(P.v)
        norm_speed = min(1.0, speed / 25.0)

        pts = getattr(self, 'nav_route_pts', None)
        if not pts or len(pts) < 2:
            pts = [(P.x, P.y), (P.x + math.cos(P.heading) * 10, P.y + math.sin(P.heading) * 10)]

        wp = pts[min(1, len(pts) - 1)]
        dx = wp[0] - P.x
        dy = wp[1] - P.y
        dist_wp = math.hypot(dx, dy)
        norm_dist_wp = min(1.0, dist_wp / 50.0)

        target_angle = math.atan2(dy, dx)
        bearing_wp = (target_angle - P.heading + math.pi) % (2.0 * math.pi) - math.pi
        norm_bearing_wp = bearing_wp / math.pi

        turn_angle = 0.0
        if len(pts) >= 3:
            wp2 = pts[min(2, len(pts) - 1)]
            angle2 = math.atan2(wp2[1] - wp[1], wp2[0] - wp[0])
            turn_angle = (angle2 - target_angle + math.pi) % (2.0 * math.pi) - math.pi
        norm_turn_angle = turn_angle / math.pi

        res = self.city.nearest_path(P.x, P.y, P.heading, P.z, maxd=10.0)
        if res is not None:
            lane, s_pos, _ = res
            lane_pt = lane.pose(s_pos)
            lane_hd = lane_pt[2]
            hd_err = (lane_hd - P.heading + math.pi) % (2.0 * math.pi) - math.pi
            norm_hd_err = hd_err / math.pi
            c_h, s_h = math.cos(lane_hd), math.sin(lane_hd)
            lat_off = -(P.x - lane_pt[0]) * s_h + (P.y - lane_pt[1]) * c_h
            norm_lat_off = float(np.clip(lat_off / 5.0, -1.0, 1.0))
            limit = lane.limit
        else:
            norm_hd_err = norm_bearing_wp
            norm_lat_off = 0.0
            limit = 11.1
        norm_limit = min(1.0, limit / 30.0)

        sig_state_val = 0.0
        norm_stop_dist = 1.0
        if res is not None:
            lane, s_pos, _ = res
            if getattr(lane, 'signal', None) and getattr(lane, 'stop_s', None) is not None:
                d_stop = lane.stop_s - s_pos
                if 0.0 <= d_stop < 60.0:
                    norm_stop_dist = d_stop / 60.0
                    st = lane.signal.state(lane.axis, self.sim_time)
                    if st == 'G':
                        sig_state_val = 1.0
                    elif st == 'Y':
                        sig_state_val = 0.2
                    elif st == 'R':
                        sig_state_val = -1.0

        cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
        veh_ahead_dist = 60.0
        rel_speed = 0.0
        nearby_cars = [c for c in getattr(self.traffic, 'cars', []) if abs(c.x - P.x) < 55.0 and abs(c.y - P.y) < 55.0]
        for car in nearby_cars:
            cdx, cdy = car.x - P.x, car.y - P.y
            fwd = cdx * cos_h + cdy * sin_h
            lat = -cdx * sin_h + cdy * cos_h
            if 0.5 < fwd < veh_ahead_dist and abs(lat) < 2.5:
                veh_ahead_dist = fwd
                rel_speed = car.v - P.v
        norm_veh_dist = veh_ahead_dist / 60.0
        norm_rel_speed = float(np.clip(rel_speed / 20.0, -1.0, 1.0))

        ped_ahead_dist = 40.0
        nearby_peds = [p for p in getattr(self.peds, 'peds', []) if abs(p.x - P.x) < 40.0 and abs(p.y - P.y) < 40.0]
        for ped in nearby_peds:
            pdx, pdy = ped.x - P.x, ped.y - P.y
            fwd = pdx * cos_h + pdy * sin_h
            lat = -pdx * sin_h + pdy * cos_h
            if 0.5 < fwd < ped_ahead_dist and abs(lat) < 3.0:
                ped_ahead_dist = fwd
        norm_ped_dist = ped_ahead_dist / 40.0

        rays = []
        angle_step = (2.0 * math.pi) / 16
        for i in range(16):
            ray_angle = P.heading + i * angle_step
            ray_dx, ray_dy = math.cos(ray_angle), math.sin(ray_angle)
            min_d = 50.0
            for car in nearby_cars:
                c_dx, c_dy = car.x - P.x, car.y - P.y
                proj = c_dx * ray_dx + c_dy * ray_dy
                if 0.5 < proj < min_d:
                    if abs(-c_dx * ray_dy + c_dy * ray_dx) < (car.width * 0.8):
                        min_d = proj
            for ped in nearby_peds:
                p_dx, p_dy = ped.x - P.x, ped.y - P.y
                proj = p_dx * ray_dx + p_dy * ray_dy
                if 0.5 < proj < min_d:
                    if abs(-p_dx * ray_dy + p_dy * ray_dx) < 1.0:
                        min_d = proj
            rays.append(min_d / 50.0)

        obs = np.array([
            norm_speed, norm_hd_err, norm_lat_off, norm_dist_wp, norm_bearing_wp,
            norm_turn_angle, norm_limit, sig_state_val, norm_stop_dist,
            norm_veh_dist, norm_rel_speed, norm_ped_dist, *rays
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------ one step
    def _step(self, steer, thr, brk, dt):
        self.sim_time += dt
        if self.crashed:
            if self.sim_time - self.crash_time >= Config.CRASH_COOLDOWN_SEC:
                self._respawn()
            return
        if self.shield > 0.0:
            self.shield = max(0.0, self.shield - dt)
        P = self.player

        # Manual controls ALWAYS override and cancel unparking
        if not getattr(self, 'auto_drive_active', False):
            self.unparking = False
            self.unparking_blocked = False

        if getattr(self, 'unparking', False) and (thr > 0.2 or abs(steer) > 0.2):
            self.unparking = False
            self.unparking_blocked = False
            self.auto_drive_active = False
            P.brake = 0.0

        P.allow_reverse = getattr(self, 'unparking', False) or (not getattr(self, 'auto_drive_active', False))
        if getattr(self, 'unparking', False):
            self._unpark_step(dt)
            P.update_z(self.city)
            self.step_count += 1
            if self.step_count % 2 == 0:
                self.traffic.update(dt * 2.0, self.sim_time, P, self.peds.peds)
                self.peds.update(dt * 2.0, self.sim_time, P, self.traffic)
            d = abs(P.v) * dt
            self.total_distance += d
            self.run_distance += d
            self.best_run = max(self.best_run, self.run_distance)
            reason = self._check_collisions()
            if reason:
                self._crash(reason)
            return
        P.update(steer, thr, brk, dt, self.city)
        P.update_z(self.city)
        # (Airplane update removed)
        self.step_count += 1
        if self.step_count % 2 == 0:
            self.traffic.update(dt * 2.0, self.sim_time, P, self.peds.peds)
            self.peds.update(dt * 2.0, self.sim_time, P, self.traffic)
        d = abs(P.v) * dt
        self.total_distance += d
        self.run_distance += d
        self.best_run = max(self.best_run, self.run_distance)
        reason = self._check_collisions()
        if reason:
            self._crash(reason)
        # Check mission arrival at landmark finish bay
        if getattr(self.city, 'destinations', None):
            d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            dest = self.city.destinations[d_idx]
            finish_pt = dest.get('finish_pt', (dest['x'], dest['y']))
            d_finish = math.hypot(P.x - finish_pt[0], P.y - finish_pt[1])
            if d_finish <= 1.5 and abs(P.v) < 0.8:
                if not getattr(self, 'mission_completed', False) or getattr(self, '_completed_dest_idx', -1) != d_idx:
                    self.mission_completed = True
                    self._completed_dest_idx = d_idx
                    self.mission_complete_t = self.sim_time
                    self.shield = max(self.shield, 6.0)
                    self._warn(f"★ MISSION ACCOMPLISHED! ARRIVED AT {dest['name'].upper()}! ★")

    # ------------------------------------------------------------- cockpit
    def _light_k(self):
        night = self.renderer.pal['night']
        return 0.45 if night else (0.75 if self.renderer.pal_name == "SUNSET" else 1.0)

    def _render_cockpit(self):
        kind = self.player.spec['kind']
        if kind == 'car':
            self._cockpit_car()
        elif kind == 'auto':
            self._cockpit_auto()
        else:
            self._cockpit_bike()

    def _speed_gauge(self, cx, y):
        s = self.screen
        P = self.player
        spd = abs(P.v) * 3.6
        t = self.font_large.render(f"{spd:3.0f}", True, (150, 240, 255))
        s.blit(t, (cx - t.get_width() // 2, y))
        s.blit(self.font_small.render("km/h", True, (120, 150, 165)), (cx - 14, y + 24))
        if P.cruise_on:
            c = self.font_main.render(f"CRUISE {P.cruise_v * 3.6:.0f}", True, (90, 255, 140))
            s.blit(c, (cx + 90, y + 4))

    def _cockpit_car(self):
        s = self.screen
        W, H = self.width, self.height
        P = self.player
        k = self._light_k()
        dash = (int(24 * k) + 6, int(26 * k) + 6, int(30 * k) + 8)
        pygame.draw.polygon(s, dash, [(0, 0), (80, 0), (30, H), (0, H)])
        pygame.draw.polygon(s, dash, [(W, 0), (W - 80, 0), (W - 30, H), (W, H)])
        pygame.draw.polygon(s, dash, [(0, 0), (W, 0), (W, 22), (0, 22)])
        hood = tuple(int(c * (0.30 + 0.45 * k)) for c in P.paint)
        hood_d = shade(hood, 0.45)
        pygame.draw.polygon(s, hood_d, [(260, H - 150), (W - 260, H - 150), (W - 40, H), (40, H)])
        pygame.draw.polygon(s, hood, [(290, H - 150), (W - 290, H - 150), (W - 150, H - 112), (150, H - 112)])
        pygame.draw.line(s, shade(hood, 1.4), (290, H - 150), (W - 290, H - 150), 2)
        pygame.draw.polygon(s, dash, [(0, H - 110), (W, H - 110), (W, H), (0, H)])
        pygame.draw.rect(s, shade(dash, 1.5), (0, H - 112, W, 3))
        cx, cy = W // 2, H - 56
        ang = -P.steer_in * 1.25
        r = 78
        for a0 in (0.0, 2.1, 4.2):
            a = ang + a0
            pygame.draw.line(s, (12, 12, 14), (cx, cy), (cx + math.sin(a) * r, cy - math.cos(a) * r * 0.42), 7)
        pygame.draw.ellipse(s, (14, 14, 16), (cx - r, cy - int(r * 0.45), r * 2, int(r * 0.9)), 9)
        pygame.draw.ellipse(s, (60, 64, 72), (cx - r + 3, cy - int(r * 0.45) + 3, r * 2 - 6, int(r * 0.9) - 6), 2)
        pygame.draw.circle(s, (30, 32, 36), (cx, cy), 14)
        self._speed_gauge(cx - 160, H - 100)

    def _cockpit_auto(self):
        s = self.screen
        W, H = self.width, self.height
        P = self.player
        k = self._light_k()
        frame = (int(40 * k) + 8, int(40 * k) + 8, int(44 * k) + 8)
        pygame.draw.polygon(s, frame, [(0, 0), (44, 0), (14, H), (0, H)])
        pygame.draw.polygon(s, frame, [(W, 0), (W - 44, 0), (W - 14, H), (W, H)])
        pygame.draw.rect(s, tuple(int(c * k) for c in (230, 190, 20)), (0, 0, W, 16))
        nose = tuple(int(c * (0.35 + 0.5 * k)) for c in P.paint)
        pygame.draw.polygon(s, shade(nose, 0.6), [(W // 2 - 190, H), (W // 2 + 190, H), (W // 2 + 70, H - 120), (W // 2 - 70, H - 120)])
        pygame.draw.polygon(s, nose, [(W // 2 - 150, H), (W // 2 + 150, H), (W // 2 + 56, H - 112), (W // 2 - 56, H - 112)])
        # handlebar
        cx, cy = W // 2, H - 52
        a = P.steer_in * 0.5
        dx, dy = math.cos(a) * 160, math.sin(a) * 40
        pygame.draw.line(s, (16, 16, 18), (cx - dx, cy - dy), (cx + dx, cy + dy), 9)
        for sg in (-1, 1):
            pygame.draw.circle(s, (24, 24, 28), (int(cx + sg * dx), int(cy + sg * dy)), 15)
            pygame.draw.circle(s, (70, 70, 76), (int(cx + sg * dx), int(cy + sg * dy)), 15, 2)
        pygame.draw.circle(s, (30, 32, 38), (cx, cy - 6), 38)
        pygame.draw.circle(s, (90, 96, 106), (cx, cy - 6), 38, 2)
        self._speed_gauge(cx, cy - 26)

    def _cockpit_bike(self):
        s = self.screen
        W, H = self.width, self.height
        P = self.player
        k = self._light_k()
        tank = tuple(int(c * (0.30 + 0.5 * k)) for c in P.paint)
        pygame.draw.polygon(s, shade(tank, 0.5), [(W // 2 - 200, H), (W // 2 + 200, H), (W // 2 + 90, H - 118), (W // 2 - 90, H - 118)])
        pygame.draw.polygon(s, tank, [(W // 2 - 160, H), (W // 2 + 160, H), (W // 2 + 70, H - 108), (W // 2 - 70, H - 108)])
        pygame.draw.line(s, shade(tank, 1.5), (W // 2 - 60, H - 100), (W // 2 + 60, H - 100), 2)
        a = P.steer_in * 0.55
        cx, cy = W // 2, H - 100
        dx, dy = math.cos(a) * 250, math.sin(a) * 38
        pygame.draw.line(s, (14, 14, 16), (cx - dx, cy - dy), (cx + dx, cy + dy), 10)
        for sg in (-1, 1):
            gx, gy = cx + sg * dx, cy + sg * dy
            pygame.draw.circle(s, (22, 22, 26), (int(gx), int(gy)), 17)
            pygame.draw.circle(s, (80, 80, 86), (int(gx), int(gy)), 17, 2)
            # side mirror stalk
            pygame.draw.line(s, (20, 20, 22), (gx, gy), (gx + sg * 24, gy - 120), 5)
            pygame.draw.ellipse(s, (30, 34, 40), (gx + sg * 24 - 26, gy - 150, 52, 34))
            pygame.draw.ellipse(s, tuple(int(c * k) for c in (120, 150, 190)), (gx + sg * 24 - 22, gy - 146, 44, 26))
        pygame.draw.circle(s, (24, 26, 30), (cx, cy - 12), 36)
        pygame.draw.circle(s, (90, 96, 106), (cx, cy - 12), 36, 2)
        self._speed_gauge(cx, cy - 32)

    def _draw_mirror(self):
        s = self.screen
        W = self.width
        mr = pygame.Rect(W // 2 - 175, 126, 350, 112)
        self._mirror_tick += 1
        if self._mirror_tick % 3 == 0 or not hasattr(self, '_mirror_ok'):
            self.renderer.render_mirror(self, self.mirror_surf)
            self._mirror_ok = True
        pygame.draw.rect(s, (14, 16, 20), mr, border_radius=14)
        s.blit(self.mirror_surf, (mr.x + 10, mr.y + 7))
        pygame.draw.rect(s, (96, 102, 112), mr, 3, border_radius=14)
        s.blit(self.font_small.render("REAR VIEW", True, (200, 210, 220)), (mr.x + 138, mr.bottom - 14))

        # Rear Camera & Side Blind-Spot Monitoring HUD
        bs = getattr(self, 'blind_spot', None)
        l_col = (60, 240, 100) if (not bs or bs.left_clear) else (255, 120, 50)
        l_text = "L: CLEAR" if (not bs or bs.left_clear) else "L: ALERT"
        r_col = (60, 240, 100) if (not bs or bs.right_clear) else (255, 120, 50)
        r_text = "R: CLEAR" if (not bs or bs.right_clear) else "R: ALERT"
        s.blit(self.font_small.render(l_text, True, l_col), (mr.x + 14, mr.bottom - 14))
        s.blit(self.font_small.render(r_text, True, r_col), (mr.right - 62, mr.bottom - 14))

    def _draw_reverse_pip(self):
        s = self.screen
        W, H = self.width, self.height
        self.renderer.render_rear_pip(self, self.pip_surf)
        r = pygame.Rect(W // 2 - 180, H - 260, 360, 210)
        pygame.draw.rect(s, (10, 12, 16), r, border_radius=10)
        s.blit(self.pip_surf, (r.x + 10, r.y + 10))
        pygame.draw.rect(s, (255, 190, 40), r, 3, border_radius=10)
        cxm = r.centerx
        for (x0, y0, x1, y1, c) in ((cxm - 70, r.bottom - 12, cxm - 30, r.y + 70, (60, 255, 90)),
                                    (cxm + 70, r.bottom - 12, cxm + 30, r.y + 70, (60, 255, 90)),
                                    (cxm - 100, r.bottom - 12, cxm - 44, r.y + 100, (255, 210, 40)),
                                    (cxm + 100, r.bottom - 12, cxm + 44, r.y + 100, (255, 210, 40))):
            pygame.draw.line(s, c, (x0, y0), (x1, y1), 2)
        s.blit(self.font_main.render("REVERSE CAMERA", True, (255, 220, 120)), (r.x + 12, r.y + 12))

        # Rear Cross-Traffic Alert (RCTA) & Obstacle Hazard Banner
        is_blocked = getattr(self, 'unparking_blocked', False)
        block_msg = getattr(self, 'unparking_blocked_reason', '')
        if not is_blocked and self.player.v < -0.2:
            is_blocked, block_msg = self._check_rear_hazards()
        if is_blocked:
            warn_banner = pygame.Surface((340, 24), pygame.SRCALPHA)
            warn_banner.fill((220, 25, 25, 225))
            s.blit(warn_banner, (r.x + 10, r.y + 36))
            s.blit(self.font_small.render(f"⚠ YIELD: {block_msg}", True, (255, 255, 255)), (r.x + 16, r.y + 40))

    # ----------------------------------------------------------------- HUD
    def _render_hud(self, obs):
        s = self.screen
        W, H = self.width, self.height
        P = self.player
        panel = pygame.Surface((W, 122), pygame.SRCALPHA)
        panel.fill((14, 17, 24, 190 if self.view_mode != "FIRST_PERSON" else 150))
        s.blit(panel, (0, 0))
        bot = pygame.Surface((W, 28), pygame.SRCALPHA)
        bot.fill((14, 17, 22, 200))
        s.blit(bot, (0, H - 28))
        if getattr(self, 'auto_drive_active', False):
            mode_text = "[AUTO DRIVE]"
            mode_col = (0, 240, 255)
        elif self.ai_mode:
            mode_text = "[AUTOPILOT]"
            mode_col = (230, 80, 255)
        else:
            mode_text = "[MANUAL]"
            mode_col = (80, 230, 140)
        s.blit(self.font_large.render(mode_text, True, mode_col), (16, 10))
        vlabel = {"FIRST_PERSON": "1ST PERSON", "CHASE": "CHASE CAM", "TOP_DOWN": "TOP DOWN"}[self.view_mode]
        if getattr(self, 'rear_on', False):
            vlabel = "REAR CAMERA"
        s.blit(self.font_main.render(f"VIEW {vlabel} [V/B]  TIME {self.renderer.pal_name} [T]", True, (240, 210, 80)), (150, 14))
        spd = abs(P.v) * 3.6
        sp = self.font_speed.render(f"{spd:3.0f}", True, (245, 250, 255))
        s.blit(sp, (16, 36))
        s.blit(self.font_main.render("km/h" + ("  REV" if P.v < -0.3 else ""), True, (170, 190, 205)), (100, 62))
        code, lim = self.city.surface_at(P.x, P.y, P.lvl)
        if code in (1, 2) and lim:
            over = spd > lim + 3
            blink = over and (int(time.time() * 3) % 2 == 0)
            col = (255, 40, 40) if not blink else (255, 190, 190)
            pygame.draw.circle(s, (250, 250, 250), (212, 62), 26)
            pygame.draw.circle(s, col, (212, 62), 26, 5)
            t = self.font_large.render(str(lim), True, (10, 10, 10))
            s.blit(t, (212 - t.get_width() // 2, 62 - t.get_height() // 2))
        else:
            pygame.draw.circle(s, (120, 120, 120), (212, 62), 26, 2)
            t = self.font_small.render("OFF", True, (150, 150, 150))
            s.blit(t, (212 - t.get_width() // 2, 62 - t.get_height() // 2))
        if P.cruise_on:
            s.blit(self.font_main.render(f"CRUISE ON  {P.cruise_v * 3.6:.0f} km/h   [+/-]", True, (90, 255, 140)), (16, 90))
        else:
            s.blit(self.font_main.render("CRUISE OFF  [C]", True, (130, 140, 150)), (16, 90))
        self.slider_speed.draw(s, self.font_main)
        self.slider_steer.draw(s, self.font_main)
        s.blit(self.font_small.render(f"{P.spec['label']} [X]   seed {self.seed}", True, (170, 200, 230)), (300, 108))
        x0 = 520
        s.blit(self.font_small.render("THR", True, (190, 200, 210)), (x0, 26))
        s.blit(self.font_small.render("BRK", True, (190, 200, 210)), (x0 + 34, 26))
        for i, (val, col) in enumerate(((P.throttle, (80, 220, 120)), (P.brake, (240, 90, 80)))):
            r = pygame.Rect(x0 + i * 34, 40, 24, 54)
            pygame.draw.rect(s, (36, 42, 52), r)
            hh = int(val * 52)
            pygame.draw.rect(s, col, (r.x, r.bottom - hh, r.w, hh))
            pygame.draw.rect(s, (110, 120, 135), r, 1)
        bx, by, bw = x0 + 80, 50, 110
        s.blit(self.font_small.render("STEERING", True, (190, 200, 210)), (bx, 30))
        pygame.draw.rect(s, (36, 42, 52), (bx, by, bw, 12))
        pygame.draw.line(s, (150, 160, 170), (bx + bw // 2, by), (bx + bw // 2, by + 12), 1)
        ix = bx + bw // 2 + int(P.steer_in * (bw // 2 - 4))
        pygame.draw.rect(s, (0, 215, 255), (ix - 3, by - 2, 6, 16))
        s.blit(self.font_small.render(f"wheel angle {math.degrees(P.steer_angle):+.1f} deg", True, (150, 165, 180)), (bx, by + 20))
        # 1. AUTO DRIVE BUTTON
        is_ad = getattr(self, 'auto_drive_active', False)
        ad_col = (0, 180, 220) if is_ad else (20, 32, 46)
        ad_border = (255, 255, 255) if is_ad else (0, 160, 220)
        pygame.draw.rect(s, ad_col, self.btn_auto_drive, border_radius=4)
        pygame.draw.rect(s, ad_border, self.btn_auto_drive, 1, border_radius=4)
        ad_txt = "[AUTO DRIVE ON]" if is_ad else "[AUTO DRIVE OFF]"
        ad_t = self.font_small.render(ad_txt, True, (255, 255, 255) if is_ad else (0, 220, 255))
        s.blit(ad_t, (self.btn_auto_drive.centerx - ad_t.get_width() // 2, self.btn_auto_drive.centery - ad_t.get_height() // 2))

        # 2. PAUSE / RESUME BUTTON
        is_p = getattr(self, 'is_paused', False)
        ps_col = (210, 110, 20) if is_p else (26, 32, 42)
        ps_border = (255, 200, 60) if is_p else (90, 110, 130)
        pygame.draw.rect(s, ps_col, self.btn_pause, border_radius=4)
        pygame.draw.rect(s, ps_border, self.btn_pause, 1, border_radius=4)
        ps_txt = "RESUME [TAB]" if is_p else "PAUSE [TAB]"
        ps_t = self.font_small.render(ps_txt, True, (255, 255, 255) if is_p else (200, 210, 225))
        s.blit(ps_t, (self.btn_pause.centerx - ps_t.get_width() // 2, self.btn_pause.centery - ps_t.get_height() // 2))

        # Flyover badge (placed to the right of the buttons)
        if P.z > 0.5:
            badge = self.font_main.render(f"  ON FLYOVER  +{P.z:.1f} m  ", True, (255, 255, 255))
            bg = pygame.Surface((badge.get_width(), badge.get_height() + 4), pygame.SRCALPHA)
            bg.fill((120, 60, 220, 210))
            s.blit(bg, (250, 126))
            s.blit(badge, (250, 128))

        # 3. AUTO DRIVE ACTIVE STATUS CARD
        if is_ad:
            d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            d_name = self.city.destinations[d_idx]['name']
            rem_d = getattr(self, 'nav_dist', 0.0)
            eta_s = getattr(self, 'nav_eta', 0.0)
            spd_kmh = abs(P.v) * 3.6

            ac_w, ac_h = 320, 68
            ac_x, ac_y = 16, 156
            ac_surf = pygame.Surface((ac_w, ac_h), pygame.SRCALPHA)
            ac_surf.fill((10, 24, 32, 225))
            s.blit(ac_surf, (ac_x, ac_y))
            pygame.draw.rect(s, (0, 230, 255), (ac_x, ac_y, ac_w, ac_h), 2, border_radius=4)

            dot_blink = (int(time.time() * 4) % 2) == 0
            dot_col = (0, 255, 180) if dot_blink else (0, 190, 130)
            pygame.draw.circle(s, dot_col, (ac_x + 12, ac_y + 14), 5)
            s.blit(self.font_main.render("AUTO DRIVE ACTIVE", True, (0, 240, 255)), (ac_x + 24, ac_y + 6))
            sub_info = f"To: {d_name[:12]} | {rem_d:.0f}m | ETA {eta_s:.0f}s | {spd_kmh:.0f} km/h"
            s.blit(self.font_small.render(sub_info, True, (220, 240, 255)), (ac_x + 10, ac_y + 28))

            # Active Driving Action / State (Explains to user why car is stopped or moving)
            if getattr(self, 'unparking', False):
                if getattr(self, 'unparking_blocked', False):
                    status_act = f"YIELDING: {getattr(self, 'unparking_blocked_reason', 'TRAFFIC IN REAR PATH')}"
                else:
                    status_act = "REVERSING SMOOTHLY OUT OF BAY"
            elif hasattr(self, 'auto_drive_controller') and self.auto_drive_controller.has_model:
                status_act = self.auto_drive_controller.live_telemetry.get('status', "CRUISING TO DESTINATION")
            else:
                status_act = "CRUISING TO DESTINATION" 
            stat_col = (255, 195, 45) if any(k in status_act for k in ('RED', 'WAITING', 'YIELDING', 'DECEL')) else (110, 255, 175)
            s.blit(self.font_small.render(f"ACTION: {status_act}", True, stat_col), (ac_x + 10, ac_y + 46))

            # Live RL Telemetry Data (Actor-Critic Policy Output & Value Estimate)
            if hasattr(self, 'auto_drive_controller') and self.auto_drive_controller.has_model:
                telem = self.auto_drive_controller.live_telemetry
                rl_w, rl_h = 320, 36
                rl_y = ac_y + ac_h + 4
                rl_surf = pygame.Surface((rl_w, rl_h), pygame.SRCALPHA)
                rl_surf.fill((12, 16, 24, 215))
                s.blit(rl_surf, (ac_x, rl_y))
                pygame.draw.rect(s, (100, 220, 255), (ac_x, rl_y, rl_w, rl_h), 1, border_radius=3)
                t_rl = f"RL POLICY: V(s)={telem['value']:+.1f} | π(s)={telem['steer']:+.2f} π(t)={telem['thr']:.2f} π(b)={telem['brk']:.2f}"
                s.blit(self.font_small.render(t_rl, True, (0, 240, 255)), (ac_x + 6, rl_y + 4))
                learned_count = telem.get('mistakes_learned', getattr(getattr(self.auto_drive_controller, 'crash_learner', None), 'total_mistakes_learned', 0))
                t_rew = f"Reward: {telem['cum_reward']:+.1f} | Learned Memory: {learned_count} mistakes saved"
                s.blit(self.font_small.render(t_rew, True, (160, 225, 180)), (ac_x + 6, rl_y + 18))

# (Destination picker modal removed for unobstructed windshield view)
        d_txt = self.font_small.render(f"DIST {self.run_distance:5.0f} m   BEST {self.best_run:5.0f} m   CRASHES {self.crash_count}",
                                       True, (200, 220, 245))
        d_bg = pygame.Surface((d_txt.get_width() + 10, d_txt.get_height() + 4), pygame.SRCALPHA)
        d_bg.fill((16, 20, 28, 190))
        s.blit(d_bg, (516, 100))
        s.blit(d_txt, (521, 102))
        if getattr(self.city, 'destinations', None):
            d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            dest = self.city.destinations[d_idx]
            dx, dy = dest['x'] - P.x, dest['y'] - P.y
            dist_d = math.hypot(dx, dy)
            rel_ang = (math.atan2(dy, dx) - P.heading + math.pi) % (2 * math.pi) - math.pi
            
            # Show upcoming route maneuver arrow (or bearing if open road)
            arrow = "^"
            if getattr(self, 'nav_route', None):
                pres = self.city.nearest_path(P.x, P.y, P.heading, P.z, maxd=15.0)
                cur_p = pres[0] if pres else None
                cur_idx = -1
                if cur_p:
                    for i, seg in enumerate(self.nav_route):
                        if getattr(seg, 'edge', None) == getattr(cur_p, 'edge', None) and getattr(seg, 'tdir', None) == getattr(cur_p, 'tdir', None):
                            cur_idx = i
                            break
                        elif getattr(seg, 'pid', None) == cur_p.pid:
                            cur_idx = i
                            break
                if cur_idx != -1:
                    for seg in self.nav_route[cur_idx:]:
                        if getattr(seg, 'kind', None) == 'conn':
                            m = getattr(seg, 'move', None)
                            if m == 'R':
                                arrow = ">"
                            elif m == 'L':
                                arrow = "<"
                            elif m == 'U':
                                arrow = "U"
                            else:
                                arrow = "^"
                            break
            if arrow == "^":
                if 0.3 <= rel_ang < 1.3:
                    arrow = ">"
                elif -1.3 < rel_ang <= -0.3:
                    arrow = "<"
                elif abs(rel_ang) >= 1.3:
                    arrow = "v"

            is_comp = getattr(self, 'mission_completed', False)

            # Sleek GPS card right above the minimap
            gx, gy, gw, gh = W - 190, H - 278, 176, 48
            card = pygame.Surface((gw, gh), pygame.SRCALPHA)
            card.fill((14, 18, 26, 220))
            s.blit(card, (gx, gy))
            border_col = (60, 255, 130) if is_comp else ((255, 220, 80) if dist_d < 35.0 else dest['color'])
            pygame.draw.rect(s, border_col, (gx, gy, gw, gh), 2, border_radius=4)

            t_dest = self.font_small.render(f"GPS [{d_idx + 1}]: {dest['name'][:14]}", True, border_col)
            s.blit(t_dest, (gx + 6, gy + 4))

            if is_comp:
                t_sub = self.font_small.render("★ COMPLETE! [1-7/G]", True, (80, 255, 140))
            elif dist_d < 35.0:
                t_sub = self.font_small.render("PULL INTO FINISH BAY!", True, (255, 230, 90))
            else:
                nav_d = getattr(self, 'nav_dist', 0.0)
                d_show = f"A* {nav_d:.0f}m" if nav_d > 0 else f"{dist_d:.0f}m"
                t_sub = self.font_small.render(f"{d_show} {arrow}   [1-7/G]", True, (240, 240, 245))
            s.blit(t_sub, (gx + 6, gy + 24))

            # Trigger banner warning on arrival
            if dist_d < 35.0 and getattr(self, '_last_arrived', None) != d_idx:
                self._last_arrived = d_idx
                self._warn(f"APPROACHING {dest['name']} - PULL INTO ENTRANCE!")
            elif dist_d >= 50.0 and getattr(self, '_last_arrived', None) == d_idx:
                self._last_arrived = None
        s.blit(self.font_small.render(f"FPS {int(self.clock.get_fps())}", True, (150, 170, 180)), (W - 70, H - 46))
        if self.shield > 0.0:
            s.blit(self.font_main.render(f"[SHIELD {self.shield:.1f}s]", True, (80, 240, 255)), (16, 106))
        if self.is_recording:
            blink = (int(time.time() * 3) % 2) == 0
            pygame.draw.circle(s, (255, 50, 50) if blink else (150, 30, 30), (736, 74), 6)
            s.blit(self.font_main.render(f"REC {self.recorded_frame_count}", True, (255, 90, 90)), (748, 67))
        ox, oy = W - Config.OBS_WIDTH - 16, 24
        s.blit(self.font_small.render(f"CNN SENSOR {Config.OBS_WIDTH}x{Config.OBS_HEIGHT}", True, (180, 210, 240)), (ox, oy - 14))
        s.blit(getattr(self, '_sensor_display_surf', pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))), (ox, oy))
        pygame.draw.rect(s, (0, 200, 255), (ox - 2, oy - 2, Config.OBS_WIDTH + 4, Config.OBS_HEIGHT + 4), 1)

        # Real-time CNN Camera Vision Detections (Road Boundaries, Lane Lines, Traffic Lights, Cars, Pedestrians)
        if hasattr(self, 'auto_drive_controller'):
            vis = getattr(self, 'latest_vision', None) or self.auto_drive_controller.vision_engine.detect(obs)
            # 1. Detected Road Boundaries (Left & Right Curb/Guardrail Edges)
            if getattr(vis, 'left_boundary', None) and len(vis.left_boundary) >= 2:
                pts_lb = [(ox + pt[0], oy + pt[1]) for pt in vis.left_boundary]
                pygame.draw.lines(s, (0, 230, 255), False, pts_lb, 2)
            if getattr(vis, 'right_boundary', None) and len(vis.right_boundary) >= 2:
                pts_rb = [(ox + pt[0], oy + pt[1]) for pt in vis.right_boundary]
                pygame.draw.lines(s, (0, 230, 255), False, pts_rb, 2)
            # 2. Detected Lane Center Path
            if getattr(vis, 'lane_path', None) and len(vis.lane_path) >= 2:
                pts_cp = [(ox + pt[0], oy + pt[1]) for pt in vis.lane_path]
                pygame.draw.lines(s, (80, 255, 140), False, pts_cp, 1)
                # Draw vehicle center reference crosshair at immediate hood (y=54)
                pygame.draw.line(s, (255, 255, 0), (ox + 100, oy + 50), (ox + 100, oy + 58), 2)
            # 3. Detected vehicle bounding brackets
            for (cx_b, cy_b, cw_b, ch_b) in vis.cars:
                pygame.draw.rect(s, (0, 240, 255), (ox + cx_b, oy + cy_b, cw_b, ch_b), 1)
            # 4. Detected pedestrian bounding brackets
            for (px_b, py_b, pw_b, ph_b) in vis.peds:
                pygame.draw.rect(s, (255, 200, 40), (ox + px_b, oy + py_b, pw_b, ph_b), 1)
            # 5. Detected traffic signal beacon
            if vis.signal_state == 'RED':
                pygame.draw.circle(s, (255, 40, 30), (ox + 12, oy + 12), 6)
                pygame.draw.circle(s, (255, 255, 255), (ox + 12, oy + 12), 6, 1)
                s.blit(self.font_small.render("RED LIGHT", True, (255, 80, 70)), (ox + 22, oy + 6))
            elif vis.signal_state == 'GREEN':
                pygame.draw.circle(s, (30, 240, 90), (ox + 12, oy + 12), 6)
                pygame.draw.circle(s, (255, 255, 255), (ox + 12, oy + 12), 6, 1)
                s.blit(self.font_small.render("GREEN", True, (80, 255, 120)), (ox + 22, oy + 6))
            # 6. Detected Roadside Signboards (Speed limit, Zebra cross, Stop)
            for s_box in getattr(vis, 'sign_boxes', []):
                bx, by, bw, bh, lbl = s_box
                pygame.draw.rect(s, (220, 80, 255), (ox + bx, oy + by, bw, bh), 2)
                t_s = self.font_small.render(lbl, True, (240, 150, 255))
                s.blit(t_s, (ox + bx, max(oy + 2, oy + by - 12)))
            # 6. Live Vision perception readout bar & Boundary Alert
            v_col = (255, 70, 70) if getattr(vis, 'boundary_alert', False) else (0, 230, 255)
            v_txt = self.font_small.render(vis.summary_text, True, v_col)
            v_bg = pygame.Surface((v_txt.get_width() + 8, v_txt.get_height() + 4), pygame.SRCALPHA)
            v_bg.fill((12, 18, 26, 210))
            s.blit(v_bg, (ox, oy + Config.OBS_HEIGHT + 6))
            s.blit(v_txt, (ox + 4, oy + Config.OBS_HEIGHT + 8))
        self._render_minimap(W - 190, H - 224, 176, 176)
        hint = ("W/S gas+brake  A/D steer  C cruise  V view  B rear cam  M map  1-7/G GPS  X vehicle  N start  K new city  "
                "T time  SPACE rec  P autopilot  R restart  ESC")
        s.blit(self.font_small.render(hint, True, (150, 160, 170)), (12, H - 20))
        # Mission Complete Victory Celebration Banner
        if getattr(self, 'mission_completed', False) or (self.sim_time - getattr(self, 'mission_complete_t', -999.0)) < 7.0:
            bw, bh = 560, 84
            bx = W // 2 - bw // 2
            by = 135
            ban = pygame.Surface((bw, bh), pygame.SRCALPHA)
            ban.fill((16, 26, 36, 235))
            s.blit(ban, (bx, by))
            pygame.draw.rect(s, (0, 255, 170), (bx, by, bw, bh), 3, border_radius=8)
            pygame.draw.rect(s, (255, 255, 255), (bx + 3, by + 3, bw - 6, bh - 6), 1, border_radius=6)
            
            d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
            d_name = self.city.destinations[d_idx]['name'].upper()
            t_win = self.font_banner.render("★ MISSION ACCOMPLISHED! ★", True, (80, 255, 160))
            s.blit(t_win, (W // 2 - t_win.get_width() // 2, by + 10))
            t_sub = self.font_main.render(f"SAFELY ARRIVED AT {d_name}  [Press 1-7 or G for Next Mission]", True, (240, 250, 255))
            s.blit(t_sub, (W // 2 - t_sub.get_width() // 2, by + 48))

        if self.warn_t > 0.0:
            t = self.font_large.render(self.warn_text, True, (255, 230, 120))
            s.blit(t, (W // 2 - t.get_width() // 2, 290))
        # PAUSED OVERLAY
        if getattr(self, 'is_paused', False):
            ov_p = pygame.Surface((W, H), pygame.SRCALPHA)
            ov_p.fill((0, 0, 0, 160))
            s.blit(ov_p, (0, 0))

            pb_w, pb_h = 460, 140
            pb_x, pb_y = W // 2 - pb_w // 2, H // 2 - pb_h // 2
            pb_surf = pygame.Surface((pb_w, pb_h), pygame.SRCALPHA)
            pb_surf.fill((16, 22, 32, 240))
            s.blit(pb_surf, (pb_x, pb_y))
            pygame.draw.rect(s, (255, 180, 40), (pb_x, pb_y, pb_w, pb_h), 2, border_radius=8)

            t_p = self.font_banner.render("PAUSED", True, (255, 200, 60))
            s.blit(t_p, (W // 2 - t_p.get_width() // 2, pb_y + 16))

            t_sub1 = self.font_main.render("Physics, AI, timers & Auto Drive frozen", True, (220, 225, 235))
            s.blit(t_sub1, (W // 2 - t_sub1.get_width() // 2, pb_y + 64))

            t_sub2 = self.font_small.render("Press TAB / PAUSE or click RESUME to continue  |  Camera [V/B] & Map [M] accessible", True, (170, 190, 210))
            s.blit(t_sub2, (W // 2 - t_sub2.get_width() // 2, pb_y + 98))

        if self.crashed:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((220, 30, 30, 95))
            s.blit(ov, (0, 0))
            b = self.font_banner.render(f"CRASH!  {self.crash_reason}", True, (255, 255, 255))
            s.blit(b, (W // 2 - b.get_width() // 2, int(H * 0.32)))

            # AI Post-Crash Reflection Card
            rep = getattr(self, 'latest_crash_report', None)
            if rep is not None:
                rw, rh = 540, 78
                rx, ry = W // 2 - rw // 2, int(H * 0.40) + 6
                r_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
                r_surf.fill((16, 20, 28, 235))
                s.blit(r_surf, (rx, ry))
                pygame.draw.rect(s, (0, 240, 255), (rx, ry, rw, rh), 2, border_radius=6)
                s.blit(self.font_main.render("⚡ AI CRASH REFLECTION & ONLINE LEARNING", True, (0, 240, 255)), (rx + 12, ry + 8))
                s.blit(self.font_small.render(f"Cause: {rep.diagnosis}", True, (255, 210, 160)), (rx + 12, ry + 32))
                s.blit(self.font_small.render(f"Fix: {rep.remediation} (PPO weights updated)", True, (140, 255, 180)), (rx + 12, ry + 52))

            left = max(0.0, Config.CRASH_COOLDOWN_SEC - (self.sim_time - self.crash_time))
            sub = self.font_large.render(f"Restarting in {left:.1f}s ...", True, (255, 232, 232))
            s.blit(sub, (W // 2 - sub.get_width() // 2, int(H * 0.40) + 94))

    def _render_minimap(self, x, y, w, h):
        P = self.player
        R = 150.0
        px, py = self.mp(P.x, P.y)
        rect = pygame.Rect(int(px - R), int(py - R), int(2 * R), int(2 * R))
        view = pygame.Surface((rect.w, rect.h))
        view.fill((30, 60, 34))
        clipped = rect.clip(self.map_surf.get_rect())
        if clipped.w > 0 and clipped.h > 0:
            view.blit(self.map_surf, (clipped.x - rect.x, clipped.y - rect.y), clipped)
        # Draw A* navigation route on minimap
        if getattr(self, 'nav_route_pts', None) and len(self.nav_route_pts) >= 2:
            pts_m = []
            for (wx, wy) in self.nav_route_pts:
                mx, my = self.mp(wx, wy)
                pts_m.append((int(mx - rect.x), int(my - rect.y)))
            if len(pts_m) >= 2:
                pygame.draw.lines(view, (0, 160, 255), False, pts_m, 6)
                pygame.draw.lines(view, (230, 255, 255), False, pts_m, 2)
        for dest in getattr(self.city, 'destinations', []):
            dx_m, dy_m = self.mp(dest['x'], dest['y'])
            dx_m -= rect.x
            dy_m -= rect.y
            if 0 <= dx_m < rect.w and 0 <= dy_m < rect.h:
                pygame.draw.circle(view, dest['color'], (int(dx_m), int(dy_m)), 5)
                pygame.draw.circle(view, (255, 255, 255), (int(dx_m), int(dy_m)), 6, 1)
        if getattr(self.city, 'destinations', None):
            cur_dest = self.city.destinations[self.dest_idx % len(self.city.destinations)]
            fpt = cur_dest.get('finish_pt', (cur_dest['x'], cur_dest['y']))
            fx_m, fy_m = self.mp(fpt[0], fpt[1])
            fx_m -= rect.x
            fy_m -= rect.y
            if 0 <= fx_m < rect.w and 0 <= fy_m < rect.h:
                f_color = (60, 255, 120) if getattr(self, 'mission_completed', False) else (255, 220, 60)
                pygame.draw.rect(view, f_color, (int(fx_m - 6), int(fy_m - 6), 12, 12), 2)
        for c in self.traffic.cars:
            cx, cy = self.mp(c.x, c.y)
            cx -= rect.x
            cy -= rect.y
            if 0 <= cx < rect.w and 0 <= cy < rect.h:
                pygame.draw.circle(view, c.color, (int(cx), int(cy)), 3)
        for p in self.peds.peds:
            cx, cy = self.mp(p.x, p.y)
            cx -= rect.x
            cy -= rect.y
            if 0 <= cx < rect.w and 0 <= cy < rect.h:
                pygame.draw.circle(view, (255, 60, 200), (int(cx), int(cy)), 3)
        for hd in self.city.signal_heads:
            cx, cy = self.mp(hd['stop'][0], hd['stop'][1])
            cx -= rect.x
            cy -= rect.y
            if 0 <= cx < rect.w and 0 <= cy < rect.h:
                st = hd['sig'].state(hd['axis'], self.sim_time)
                pygame.draw.circle(view, {'R': (255, 40, 30), 'Y': (255, 190, 20), 'G': (30, 235, 90)}[st], (int(cx), int(cy)), 3)
        self._draw_player_arrow(view, px - rect.x, py - rect.y, P.heading, 8)
        view = pygame.transform.smoothscale(view, (w, h))
        self.screen.blit(view, (x, y))
        pygame.draw.rect(self.screen, (0, 200, 255), (x - 1, y - 1, w + 2, h + 2), 2)

    @staticmethod
    def _draw_player_arrow(surf, x, y, h, size):
        c, s = math.cos(h), math.sin(h)
        pts = [(x + c * size * 1.5, y + s * size * 1.5),
               (x - c * size + s * size, y - s * size - c * size),
               (x - c * size * 0.4, y - s * size * 0.4),
               (x - c * size - s * size, y - s * size + c * size)]
        pygame.draw.polygon(surf, (0, 220, 255), pts)
        pygame.draw.polygon(surf, (255, 255, 255), pts, 1)

    def _render_big_map(self):
        s = self.screen
        W, H = self.width, self.height
        s.fill((10, 14, 20))

        # 1. Sidebar on the left: Landmark Options List
        side_w = 236
        side_x, side_y, side_h = 14, 44, H - 76
        side_surf = pygame.Surface((side_w, side_h), pygame.SRCALPHA)
        side_surf.fill((14, 18, 26, 235))
        s.blit(side_surf, (side_x, side_y))
        pygame.draw.rect(s, (45, 65, 85), (side_x, side_y, side_w, side_h), 1, border_radius=6)

        s.blit(self.font_main.render("LANDMARK OPTIONS", True, (0, 220, 255)), (side_x + 10, side_y + 8))
        s.blit(self.font_small.render("Press 1-7 or Click to Route:", True, (160, 175, 190)), (side_x + 10, side_y + 28))

        self._landmark_rects = []
        P = self.player
        destinations = getattr(self.city, 'destinations', [])

        row_h = (side_h - 56) // max(1, len(destinations))
        for i, dest in enumerate(destinations):
            ry = side_y + 48 + i * row_h
            r_rect = pygame.Rect(side_x + 6, ry, side_w - 12, row_h - 4)
            self._landmark_rects.append((r_rect, i))

            is_active = (i == self.dest_idx)
            dx, dy = dest['x'] - P.x, dest['y'] - P.y
            d_straight = math.hypot(dx, dy)

            if is_active:
                pygame.draw.rect(s, (24, 52, 78), r_rect, border_radius=4)
                pygame.draw.rect(s, (0, 225, 255), r_rect, 2, border_radius=4)
                col_title = (0, 240, 255)
            else:
                pygame.draw.rect(s, (20, 25, 34), r_rect, border_radius=4)
                pygame.draw.rect(s, (40, 50, 64), r_rect, 1, border_radius=4)
                col_title = (225, 230, 240)

            t_num = f"[{i + 1}]"
            s.blit(self.font_small.render(f"{t_num} {dest['name'][:14]}", True, col_title), (r_rect.x + 8, r_rect.y + 4))

            if is_active:
                route_d = getattr(self, 'nav_dist', d_straight)
                route_t = getattr(self, 'nav_eta', route_d / 14.0)
                sub_txt = f"A* {route_d:.0f}m (~{route_t:.0f}s)"
                col_sub = (100, 255, 170)
            else:
                sub_txt = f"Direct: {d_straight:.0f}m"
                col_sub = (130, 145, 160)
            s.blit(self.font_small.render(sub_txt, True, col_sub), (r_rect.x + 8, r_rect.y + 20))

        # 2. Main Map on the right
        map_x0 = side_x + side_w + 14
        map_w_avail = W - map_x0 - 16
        map_h_avail = H - 76
        mw, mh = self.map_surf.get_size()
        sc = min(map_h_avail / mh, map_w_avail / mw)
        size = (int(mw * sc), int(mh * sc))
        mp_ = pygame.transform.smoothscale(self.map_surf, size)
        ox = map_x0 + (map_w_avail - size[0]) // 2
        oy = side_y + (map_h_avail - size[1]) // 2
        s.blit(mp_, (ox, oy))

        def MP(x, y):
            a, b = self.mp(x, y)
            return (ox + a * sc, oy + b * sc)

        # Draw A* path across the city map
        if getattr(self, 'nav_route_pts', None) and len(self.nav_route_pts) >= 2:
            screen_pts = [MP(wx, wy) for (wx, wy) in self.nav_route_pts]
            if len(screen_pts) >= 2:
                pygame.draw.lines(s, (0, 170, 255), False, screen_pts, 6)
                pygame.draw.lines(s, (230, 255, 255), False, screen_pts, 3)
                for pi in range(0, len(screen_pts), max(1, len(screen_pts) // 18)):
                    pygame.draw.circle(s, (0, 240, 255), (int(screen_pts[pi][0]), int(screen_pts[pi][1])), 3)

        # Draw signals on map
        for hd in self.city.signal_heads:
            st = hd['sig'].state(hd['axis'], self.sim_time)
            q = MP(*hd['stop'])
            pygame.draw.circle(s, {'R': (255, 40, 30), 'Y': (255, 190, 20), 'G': (30, 235, 90)}[st], (int(q[0]), int(q[1])), 2)

        # Draw traffic & peds
        for c in self.traffic.cars:
            q = MP(c.x, c.y)
            pygame.draw.circle(s, c.color, (int(q[0]), int(q[1])), 2)
        for p in self.peds.peds:
            q = MP(p.x, p.y)
            pygame.draw.circle(s, (255, 60, 200), (int(q[0]), int(q[1])), 3)

        # Draw destination pins and track for clicks
        self._landmark_pins = []
        for i, dest in enumerate(destinations):
            q = MP(dest['x'], dest['y'])
            self._landmark_pins.append((q, i))
            is_active = (i == self.dest_idx)
            rad = 11 if is_active else 8
            pygame.draw.circle(s, dest['color'], (int(q[0]), int(q[1])), rad)
            pygame.draw.circle(s, (255, 255, 255), (int(q[0]), int(q[1])), rad + 1, 2)
            if is_active:
                pulse = int(14 + (int(time.time() * 4) % 3) * 2)
                pygame.draw.circle(s, (0, 240, 255), (int(q[0]), int(q[1])), pulse, 2)
            lbl = self.font_small.render(f"[{i + 1}] {dest['name']}", True, (255, 255, 230) if not is_active else (80, 255, 180))
            s.blit(lbl, (int(q[0]) + 12, int(q[1]) - 7))

        if destinations:
            cur_dest = destinations[self.dest_idx % len(destinations)]
            fpt = cur_dest.get('finish_pt', (cur_dest['x'], cur_dest['y']))
            qf = MP(fpt[0], fpt[1])
            f_color = (60, 255, 120) if getattr(self, 'mission_completed', False) else (255, 220, 60)
            pygame.draw.rect(s, f_color, (int(qf[0] - 8), int(qf[1] - 8), 16, 16), 2)

        # Player location arrow
        q = MP(self.player.x, self.player.y)
        self._draw_player_arrow(s, q[0], q[1], self.player.heading, 9)

        # Top Bar
        s.blit(self.font_large.render(f"CITY MAP  (seed {self.seed})    [1-7/Click] Landmark Options   [M] close   [K] new city",
                                      True, (230, 235, 245)), (20, 10))

        # Bottom legend
        lx = 20
        for lab, key in (("Highway", 'H'), ("Avenue", 'A'), ("Street", 'S'), ("Narrow lane", 'L'), ("One-way", 'O')):
            pygame.draw.rect(s, MAP_COLORS[key], (lx, H - 22, 18, 10))
            t = self.font_main.render(lab, True, (220, 225, 235))
            s.blit(t, (lx + 24, H - 24))
            lx += 36 + t.get_width() + 10
        pygame.draw.rect(s, (176, 96, 255), (lx, H - 22, 18, 10))
        s.blit(self.font_main.render("Flyover", True, (220, 225, 235)), (lx + 24, H - 24))

        if getattr(self, 'nav_dist', 0) > 0:
            badge_t = f"A* ROUTE: {self.nav_dist:.0f}m  (~{self.nav_eta:.0f}s)"
            s.blit(self.font_main.render(badge_t, True, (0, 240, 255)), (W - 320, H - 24))

    # ---------------------------------------------------------------- loop
    def _read_inputs(self):
        k = pygame.key.get_pressed()
        steer = thr = brk = 0.0
        if k[pygame.K_LEFT] or k[pygame.K_a]:
            steer -= 1.0
        if k[pygame.K_RIGHT] or k[pygame.K_d]:
            steer += 1.0
        if k[pygame.K_UP] or k[pygame.K_w]:
            thr = 1.0
        if k[pygame.K_DOWN] or k[pygame.K_s]:
            brk = 1.0
        if self.joystick is not None:
            try:
                ax = self.joystick.get_axis(0)
                if abs(ax) > 0.08:
                    steer = ax
                if self.joystick.get_numaxes() >= 6:
                    rt = (self.joystick.get_axis(5) + 1.0) * 0.5
                    lt = (self.joystick.get_axis(4) + 1.0) * 0.5
                    if rt > 0.05:
                        thr = rt
                    if lt > 0.05:
                        brk = lt
            except Exception:
                pass
        return steer, thr, brk

    def _warn(self, text):
        self.warn_text, self.warn_t = text, 2.2

    def _handle_key(self, k):
        if k == pygame.K_SPACE:
            self.is_recording = not self.is_recording
            print(f"[RECORDING] {'ACTIVE' if self.is_recording else 'STOPPED'} ({self.recorded_frame_count} frames)")
        elif k == pygame.K_t:
            i = TIME_ORDER.index(self.renderer.pal_name)
            self.set_time_of_day(TIME_ORDER[(i + 1) % 3])
        elif k == pygame.K_p:
            if self.ai_model is not None:
                self.ai_mode = not self.ai_mode
                if self.ai_mode:
                    self.auto_drive_active = False  # Mutual exclusion
            else:
                self._warn("No model loaded (use --model path.pt)")
        elif k == pygame.K_o:
            self.toggle_auto_drive()
        elif k in (pygame.K_TAB, pygame.K_PAUSE):
            self.toggle_pause()
        elif k == pygame.K_v:
            self.rear_on = False
            self.view_mode = VIEW_ORDER[(VIEW_ORDER.index(self.view_mode) + 1) % 3]
        elif k == pygame.K_b:
            self.rear_on = not getattr(self, 'rear_on', False)
            self._warn("Rear camera ON" if self.rear_on else "Rear camera OFF")
        elif k == pygame.K_m:
            self.show_map = not self.show_map
        elif k in (pygame.K_1, pygame.K_KP1):
            self.select_destination(0)
        elif k in (pygame.K_2, pygame.K_KP2):
            self.select_destination(1)
        elif k in (pygame.K_3, pygame.K_KP3):
            self.select_destination(2)
        elif k in (pygame.K_4, pygame.K_KP4):
            self.select_destination(3)
        elif k in (pygame.K_5, pygame.K_KP5):
            self.select_destination(4)
        elif k in (pygame.K_6, pygame.K_KP6):
            self.select_destination(5)
        elif k in (pygame.K_7, pygame.K_KP7):
            self.select_destination(6)
        elif k == pygame.K_g:
            if getattr(self.city, 'destinations', None):
                self.select_destination((getattr(self, 'dest_idx', 0) + 1) % len(self.city.destinations))
        elif k == pygame.K_n:
            self.spawn_idx = (self.spawn_idx + 1) % len(self.city.spawns)
            self._respawn()
            self._warn("Start: " + self.spawn_label)
        elif k == pygame.K_k:
            self.new_city()
        elif k == pygame.K_x:
            i = PLAYER_VEHICLES.index(self.vehicle)
            self.vehicle = PLAYER_VEHICLES[(i + 1) % len(PLAYER_VEHICLES)]
            self._respawn()
            self._warn("Vehicle: " + VEHICLES[self.vehicle]['label'])
        elif k == pygame.K_r:
            self._respawn()
        elif k == pygame.K_c:
            was_on = self.player.cruise_on
            on = self.player.toggle_cruise()
            if on:
                self._warn(f"Cruise control ON  {self.player.cruise_v * 3.6:.0f} km/h")
            elif was_on:
                self._warn("Cruise control OFF")
            else:
                self._warn("Cruise needs more than 15 km/h")
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_RIGHTBRACKET):
            self.player.adjust_cruise(Config.CRUISE_STEP)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_LEFTBRACKET):
            self.player.adjust_cruise(-Config.CRUISE_STEP)

    def run(self, max_frames=None, script=None):
        running = True
        print("\n=======================================================")
        print("  CITY DRIVING SIMULATOR  -  start:", self.spawn_label)
        print("  Vehicle:", self.player.spec['label'], "  Mode:", "AUTOPILOT" if self.ai_mode else "MANUAL")
        print("  W/S gas+brake  A/D steer  G GPS dest  N visit spawn  M map  B rear cam  X vehicle  ESC quit")
        print("=======================================================\n")
        acc = 0.0
        dt = Config.PHYS_DT
        self.rear_on = getattr(self, 'rear_on', False)
        while running:
            if self.headless:
                self.clock.tick()
                frame_dt = dt
            else:
                frame_dt = min(self.clock.tick(self.target_fps) / 1000.0, 0.1)
            self.frame_index += 1
            if self.warn_t > 0.0:
                self.warn_t -= frame_dt
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if self.btn_auto_drive.collidepoint(ev.pos):
                            self.toggle_auto_drive()
                        elif self.btn_pause.collidepoint(ev.pos):
                            self.toggle_pause()
                        elif self.show_map:
                            self._handle_map_click(ev.pos)
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (4, 5) and self.view_mode == "TOP_DOWN":
                        self.renderer.td_scale = clamp(self.renderer.td_scale * (1.12 if ev.button == 4 else 0.89), 1.5, 14.0)
                    if self.slider_speed.handle_event(ev):
                        Config.MAX_SPEED = self.slider_speed.val / 3.6
                    if self.slider_steer.handle_event(ev):
                        Config.STEER_SENSITIVITY = self.slider_steer.val
                elif ev.type == pygame.MOUSEWHEEL and self.view_mode == "TOP_DOWN":
                    self.renderer.td_scale = clamp(self.renderer.td_scale * (1.12 ** ev.y), 1.5, 14.0)
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self._handle_key(ev.key)

            steer, thr, brk = self._read_inputs()
            if script is not None:
                steer, thr, brk = script(self)

            # Manual override cancels Auto Drive (with 0.6s grace period and deadband to avoid accidental cancellation)
            if getattr(self, 'auto_drive_active', False):
                time_since_engage = self.sim_time - getattr(self, '_auto_drive_engage_t', 0.0)
                if time_since_engage > 0.6:
                    if abs(steer) > 0.35 or thr > 0.35 or brk > 0.35:
                        self.auto_drive_active = False
                        self.unparking = False
                        self.unparking_blocked = False
                        self.player.brake = 0.0
                        self._warn("Auto Drive CANCELLED by manual override")

            # Forward throttle directly aborts reverse unparking at any time
            if getattr(self, 'unparking', False) and thr > 0.25:
                self.unparking = False
                self.unparking_blocked = False
                self.auto_drive_active = False
                self.player.brake = 0.0
                self._warn("Manual Override: Full control restored")

            if getattr(self, 'auto_drive_active', False) and not self.crashed and not getattr(self, 'is_paused', False):
                if getattr(self, 'unparking', False):
                    steer = getattr(self.player, 'steer_in', 0.0)
                    thr = 0.0
                    brk = 0.0
                else:
                    # Check destination arrival
                    d_idx = getattr(self, 'dest_idx', 0) % len(self.city.destinations)
                    dest = self.city.destinations[d_idx]
                    finish_pt = dest.get('finish_pt', (dest['x'], dest['y']))
                    d_finish = math.hypot(self.player.x - finish_pt[0], self.player.y - finish_pt[1])

                    if getattr(self, 'mission_completed', False) or (d_finish <= 1.3 and abs(self.player.v) < 0.6):
                        steer = 0.0
                        thr = 0.0
                        brk = 1.0
                        self.auto_drive_active = False
                        self.mission_completed = True
                        self._warn(f"★ SAFELY PARKED AT {dest['name'].upper()}! ★")
                    else:
                        rl_obs = self.get_rl_observation()
                        steer, thr, brk = self.auto_drive_controller.step(self, rl_obs)

            elif self.ai_mode and self.ai_model is not None and not self.crashed and not getattr(self, 'is_paused', False):
                a_s, a_t = predict_control(self.ai_model, self._obs, self.ai_device)
                steer = a_s
                thr, brk = (a_t, 0.0) if a_t >= 0 else (0.0, -a_t)

            if not getattr(self, 'is_paused', False):
                acc += frame_dt
                n = 0
                while acc >= dt and n < 6:
                    try:
                        self._step(steer, thr, brk, dt)
                    except Exception as e:
                        print(f"[SIM STEP ERROR RECOVERED]: {e}")
                    acc -= dt
                    n += 1
                if n == 6:
                    acc = 0.0
                try:
                    self.update_navigation()
                except Exception as e:
                    print(f"[NAV UPDATE ERROR RECOVERED]: {e}")

            # ---- render the world
            if self.view_mode == "TOP_DOWN" and not self.rear_on:
                self.renderer.render_top(self)
            elif self.rear_on:
                self.renderer.render_3d(self, "REAR")
            else:
                self.renderer.render_3d(self, self.view_mode)
            obs = self.get_observation()
            self._obs = obs
            if self.is_recording and not self.crashed and not getattr(self, 'is_paused', False) and abs(self.player.v) > 0.5 \
                    and (self.frame_index % Config.RECORD_INTERVAL) == 0:
                self._save_recorded_frame(obs)

            self.screen.blit(self.world_surface, (0, 0))
            first = (self.view_mode == "FIRST_PERSON" and not self.rear_on)
            if first:
                self._render_cockpit()
                self._draw_mirror()
            if self.show_map:
                self._render_big_map()
            else:
                if self.rear_on:
                    s_ = self.screen
                    pygame.draw.rect(s_, (255, 190, 40), (6, 6, self.width - 12, self.height - 12), 3)
                if self.player.v < -0.3 and not self.rear_on and self.view_mode in ("FIRST_PERSON", "CHASE"):
                    self._draw_reverse_pip()
                self._render_hud(obs)
            if not self.headless:
                pygame.display.flip()
            if max_frames is not None and self.frame_index >= max_frames:
                print(f"[SIM] Reached frame limit {max_frames}.")
                break
        self._print_exit_summary()
        pygame.quit()

    def _print_exit_summary(self):
        rt = time.time() - self.sim_start
        print("\n=======================================================")
        print("                SIMULATION SESSION SUMMARY")
        print("=======================================================")
        print(f"  City seed:           {self.seed}")
        print(f"  Runtime:             {rt:.1f} s")
        print(f"  Frames:              {self.frame_index}")
        print(f"  Distance driven:     {self.total_distance:.0f} m")
        print(f"  Best run:            {self.best_run:.0f} m")
        print(f"  Crashes:             {self.crash_count}")
        print(f"  Dataset frames:      {self.recorded_frame_count}")
        print("=======================================================\n")


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="City Driving Simulator - random cities, flyovers")
    ap.add_argument("--model", type=str, default=None, help="PyTorch model (.pt/.pth) for autopilot")
    ap.add_argument("--rl-model", type=str, default=("rl_policy.pt" if os.path.exists("rl_policy.pt") else None), help="RL policy model path (.pt) for Auto Drive")
    ap.add_argument("--seed", type=int, default=None, help="City seed (omit for a new random city every run)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fps", type=int, default=Config.FPS)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--time", type=str, default="day", choices=["day", "sunset", "night"])
    ap.add_argument("--record-interval", type=int, default=Config.RECORD_INTERVAL)
    ap.add_argument("--spawn", type=int, default=0, help="Start location index (N key cycles in game)")
    ap.add_argument("--traffic", type=int, default=None, help="Number of AI vehicles (default 55)")
    ap.add_argument("--vehicle", type=str, default="sedan", choices=PLAYER_VEHICLES, help="Your vehicle")
    args = ap.parse_args()
    if args.fps:
        Config.FPS = args.fps
    Config.RECORD_INTERVAL = args.record_interval
    sim = Simulator(model_path=args.model, seed=args.seed, headless=args.headless, target_fps=Config.FPS,
                    spawn=args.spawn, traffic=args.traffic, vehicle=args.vehicle, rl_model=args.rl_model)
    sim.set_time_of_day(args.time)
    sim.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
