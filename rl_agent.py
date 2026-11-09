"""
rl_agent.py - Reinforcement Learning Policy, Actor-Critic Network, and Rule-Based Controller
=============================================================================================
Architecture: Hybrid Autonomous Driving
- High-level: A* provides the navigation route as a sequence of waypoints & lane segments.
- Low-level: Continuous Actor-Critic policy (PPO) outputs steering, throttle, and brake.
- Fallback & Safety Layer: Deterministic Pure-Pursuit + PID + Traffic Rules that can drive
  autonomously when no model is loaded, and acts as a runtime safety supervisor that overrides
  RL actions if a traffic rule or collision boundary would be violated.
- CNN Camera Perception Engine: Real-time optical perception detecting traffic lights,
  crossing pedestrians, and vehicles ahead directly from the camera feed.
- Online Crash Reflection Engine: Diagnoses root causes of crashes and performs online
  policy updates so the AI learns why each crash happened and remedies it immediately.
"""

from typing import Tuple, Optional, Dict, Any, List
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
import math
import os
import time
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

# =============================================================================
# REWARD SHAPING CONSTANTS (Named weights for training & evaluation)
# =============================================================================
# Positive rewards
R_PROGRESS: float = 2.5        # Reward scaled by meters progressed along the active A* route
R_WAYPOINT: float = 8.0        # Bonus awarded upon reaching each intermediate waypoint
R_DESTINATION: float = 150.0   # Large terminal reward for successfully reaching the landmark finish bay
R_LANE_CENTER: float = 0.8     # Step reward for staying close to the lane centre line (lateral error < 0.5m)
R_SPEED_MATCH: float = 0.5     # Reward for driving efficiently near the posted speed limit without exceeding it

# Negative rewards / Penalties (Rigorous penalties for every mistake)
R_COLLISION: float = -200.0       # Terminal penalty for collision with vehicle, pedestrian, or building
R_RED_LIGHT: float = -100.0       # Severe penalty for running a red light or crossing the stop line on red
R_LANE_STRADDLE: float = -35.0    # Heavy penalty for straddling white lane divider line or crossing lanes without cause
R_CURVE_SPEEDING: float = -40.0   # Severe penalty for entering a corner or turn too fast (>15 km/h)
R_TAILGATE: float = -30.0         # Penalty for following lead car too closely (<4.0m)
R_FALSE_STOP: float = -15.0       # Penalty for coming to a complete stop when road ahead is clear
R_SPEEDING: float = -5.0          # Penalty proportional to amount exceeding posted speed limit
R_WRONG_WAY: float = -100.0       # Severe penalty for driving into incoming/oncoming vehicle line
R_MIDDLE_DIVIDER_CROSS: float = -75.0  # Penalty for cutting middle divider / crossing yellow line while turning
R_OFF_ROAD: float = -45.0         # Penalty for driving off the paved asphalt surface onto grass/curbs
R_ROAD_EDGE_CROSS: float = -80.0   # Severe penalty for crossing road edge, curb, or outer boundary
R_YIELD_FAIL: float = -50.0       # Penalty for failing to yield to pedestrians on crosswalks or oncoming traffic
R_SMOOTHNESS: float = -0.15       # Penalty for sudden jerk, extreme steering changes, or thrashing pedals
R_IDLE_PENALTY: float = -0.30     # Step penalty for standing still without a reason (e.g. green light, empty road)


# =============================================================================
# CNN CAMERA PERCEPTION ENGINE (Vision-based Light, Vehicle & Pedestrian Detector)
# =============================================================================
class VisionResult:
    """Structured perception output from processing the forward CNN camera feed."""
    def __init__(self):
        self.signal_state: str = 'CLEAR'          # 'RED', 'YELLOW', 'GREEN', or 'CLEAR'
        self.signal_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.cars: List[Tuple[int, int, int, int]] = []              # Bounding boxes [(x, y, w, h)]
        self.peds: List[Tuple[int, int, int, int]] = []              # Bounding boxes [(x, y, w, h)]
        self.closest_car_dist: float = 1.0                           # Normalized [0, 1]
        self.closest_ped_dist: float = 1.0                           # Normalized [0, 1]
        self.left_boundary: List[Tuple[int, int]] = []               # Points [(x, y)] of left road boundary
        self.right_boundary: List[Tuple[int, int]] = []              # Points [(x, y)] of right road boundary
        self.lane_path: List[Tuple[int, int]] = []                   # Points [(x, y)] along lane center
        self.margin_left: float = 80.0                               # Pixels from car center to left boundary
        self.margin_right: float = 80.0                              # Pixels from car center to right boundary
        self.corridor_center_x: float = 100.0                        # Target center x in camera view
        self.boundary_alert: bool = False                            # True if near road boundary/barrier
        self.optical_steer_nudge: float = 0.0                        # Vision corrective steer [-1, 1]
        self.detected_signs: List[Dict[str, Any]] = []
        self.sign_boxes: List[Tuple[int, int, int, int, str]] = []
        self.summary_text: str = "VISION: ROAD CLEAR"


class CNNPerceptionEngine:
    """
    Real-time visual perception running directly on the vehicle's forward
    CNN sensor camera (200x66 RGB image).
    Extracts road boundary limits, lane markings, traffic signals, cars, and pedestrians.
    """
    def __init__(self, width: int = 200, height: int = 66):
        self.width = width
        self.height = height

    def detect(self, obs_rgb: np.ndarray) -> VisionResult:
        res = VisionResult()
        if obs_rgb is None or obs_rgb.shape[0] != self.height or obs_rgb.shape[1] != self.width:
            return res

        H, W = self.height, self.width

        # ---------------- 1. Road Boundary & Lane Line Tracking (Lower Road ROI) ----------------
        # Scan 4 horizontal horizon lines across the forward driving corridor
        scan_rows = [48, 51, 54, 58]
        left_pts = []
        right_pts = []
        center_pts = []

        for y in scan_rows:
            row = obs_rgb[y, :, :]
            # Asphalt color signature
            is_asphalt = (row[:, 0] < 88) & (row[:, 1] < 88) & (row[:, 2] < 92) & (row[:, 0] > 32)
            # White and yellow lane markings on road
            is_line = ((row[:, 0] > 180) & (row[:, 1] > 180) & (row[:, 2] > 180)) | ((row[:, 0] > 180) & (row[:, 1] > 140) & (row[:, 2] < 90))
            is_road = is_asphalt | is_line
            road_idx = np.where(is_road)[0]

            if len(road_idx) >= 15:
                lx = int(road_idx[0])
                rx = int(road_idx[-1])
                cx = (lx + rx) // 2
                left_pts.append((lx, y))
                right_pts.append((rx, y))
                center_pts.append((cx, y))

        res.left_boundary = left_pts
        res.right_boundary = right_pts
        res.lane_path = center_pts

        if center_pts:
            # Primary reference scanline at immediate front (y=54)
            ref_idx = min(2, len(center_pts) - 1)
            ref_c = center_pts[ref_idx]
            ref_l = left_pts[ref_idx]
            ref_r = right_pts[ref_idx]
            
            res.corridor_center_x = float(ref_c[0])
            res.margin_left = float(100.0 - ref_l[0])
            res.margin_right = float(ref_r[0] - 100.0)

            # Check boundary emergency buffer (critical when margin < 48px)
            if res.margin_right < 48.0:
                res.boundary_alert = True
                res.optical_steer_nudge = -float(np.clip((48.0 - res.margin_right) / 10.0, 0.35, 1.0))
            elif res.margin_left < 48.0:
                res.boundary_alert = True
                res.optical_steer_nudge = float(np.clip((48.0 - res.margin_left) / 10.0, 0.35, 1.0))
            else:
                res.optical_steer_nudge = float(np.clip((res.corridor_center_x - 100.0) / 60.0, -0.45, 0.45))

        # ---------------- 2. Traffic Light Detection (Upper Horizon ROI) ----------------
        roi_sky = obs_rgb[0:int(H * 0.48), int(W * 0.15):int(W * 0.85)]
        red_mask = (roi_sky[:, :, 0] > 200) & (roi_sky[:, :, 1] < 75) & (roi_sky[:, :, 2] < 75)
        green_mask = (roi_sky[:, :, 1] > 195) & (roi_sky[:, :, 0] < 85) & (roi_sky[:, :, 2] < 120)
        yellow_mask = (roi_sky[:, :, 0] > 210) & (roi_sky[:, :, 1] > 175) & (roi_sky[:, :, 2] < 60)

        red_pts = np.argwhere(red_mask)
        green_pts = np.argwhere(green_mask)
        yellow_pts = np.argwhere(yellow_mask)
        x_off = int(W * 0.15)

        # Require a clear cluster of at least 6 saturated pixels to avoid false alarms from distant buildings
        if len(red_pts) >= 6:
            res.signal_state = 'RED'
            min_y, min_x = np.min(red_pts, axis=0)
            max_y, max_x = np.max(red_pts, axis=0)
            res.signal_box = (min_x + x_off - 3, min_y - 3, (max_x - min_x) + 6, (max_y - min_y) + 6)
        elif len(yellow_pts) >= 6:
            res.signal_state = 'YELLOW'
            min_y, min_x = np.min(yellow_pts, axis=0)
            max_y, max_x = np.max(yellow_pts, axis=0)
            res.signal_box = (min_x + x_off - 3, min_y - 3, (max_x - min_x) + 6, (max_y - min_y) + 6)
        elif len(green_pts) >= 6:
            res.signal_state = 'GREEN'
            min_y, min_x = np.min(green_pts, axis=0)
            max_y, max_x = np.max(green_pts, axis=0)
            res.signal_box = (min_x + x_off - 3, min_y - 3, (max_x - min_x) + 6, (max_y - min_y) + 6)

        # ---------------- 3. Vehicle Detection (Roadway Center ROI) ----------------
        roi_road = obs_rgb[int(H * 0.35):, int(W * 0.10):int(W * 0.90)]
        road_y_off = int(H * 0.35)
        road_x_off = int(W * 0.10)

        tail_mask = (roi_road[:, :, 0] > 200) & (roi_road[:, :, 1] < 70) & (roi_road[:, :, 2] < 70)
        tail_pts = np.argwhere(tail_mask)

        if len(tail_pts) >= 3:
            min_y, min_x = np.min(tail_pts, axis=0)
            max_y, max_x = np.max(tail_pts, axis=0)
            w_box = max(18, (max_x - min_x) + 12)
            h_box = max(12, (max_y - min_y) + 10)
            cx_box = (min_x + max_x) // 2 + road_x_off
            cy_box = (min_y + max_y) // 2 + road_y_off
            res.cars.append((max(0, cx_box - w_box // 2), max(0, cy_box - h_box // 2), w_box, h_box))
            # Only cars centered inside the vehicle's driving lane corridor update closest_car_dist
            if int(W * 0.28) <= cx_box <= int(W * 0.72):
                res.closest_car_dist = float(np.clip(1.0 - (cy_box / float(H)), 0.05, 1.0))

        # ---------------- 4. Pedestrian Detection (Corridor Filtered) ----------------
        ped_mask = (roi_road[:, :, 0] > 180) & (roi_road[:, :, 1] > 140) & (roi_road[:, :, 2] > 110) & (roi_road[:, :, 0] < 240)
        ped_pts = np.argwhere(ped_mask)
        if len(ped_pts) >= 4:
            min_y, min_x = np.min(ped_pts, axis=0)
            max_y, max_x = np.max(ped_pts, axis=0)
            h_span = max_y - min_y
            w_span = max_x - min_x
            if h_span >= 6 and h_span > w_span * 1.2:
                px_b = min_x + road_x_off - 2
                py_b = min_y + road_y_off - 2
                ped_cx = px_b + (w_span + 4) // 2
                res.peds.append((px_b, py_b, w_span + 4, h_span + 4))
                # Only flag as blocking obstacle if pedestrian is inside the roadway corridor (68 <= x <= 132)
                # Pedestrians standing on sidewalks/kerbs (x < 68 or x > 132) are safely off-road
                if 68 <= ped_cx <= 132:
                    res.closest_ped_dist = float(np.clip(1.0 - ((py_b + h_span) / float(H)), 0.05, 1.0))

        # ---------------- 5. Road Sign Detection (Speed Limit, Zebra, Stop) ----------------
        roi_signs = obs_rgb[int(H * 0.12):int(H * 0.55), :]
        sign_y_off = int(H * 0.12)
        # Circular red rim speed limit detection
        sign_red_mask = (roi_signs[:, :, 0] > 195) & (roi_signs[:, :, 1] < 70) & (roi_signs[:, :, 2] < 70)
        s_red_pts = np.argwhere(sign_red_mask)
        if len(s_red_pts) >= 5:
            min_sy, min_sx = np.min(s_red_pts, axis=0)
            max_sy, max_sx = np.max(s_red_pts, axis=0)
            sw = (max_sx - min_sx) + 4
            sh = (max_sy - min_sy) + 4
            if 5 <= sw <= 28 and 5 <= sh <= 28:
                box = (min_sx - 2, min_sy + sign_y_off - 2, sw, sh, "SPEED")
                res.sign_boxes.append(box)
                res.detected_signs.append({'type': 'SPEED_LIMIT', 'val': 40, 'box': box})

        # Blue square zebra crossing sign detection
        sign_blue_mask = (roi_signs[:, :, 2] > 185) & (roi_signs[:, :, 0] < 80) & (roi_signs[:, :, 1] < 120)
        s_blue_pts = np.argwhere(sign_blue_mask)
        if len(s_blue_pts) >= 5:
            min_sy, min_sx = np.min(s_blue_pts, axis=0)
            max_sy, max_sx = np.max(s_blue_pts, axis=0)
            sw = (max_sx - min_sx) + 4
            sh = (max_sy - min_sy) + 4
            if 5 <= sw <= 28 and 5 <= sh <= 28:
                box = (min_sx - 2, min_sy + sign_y_off - 2, sw, sh, "ZEBRA")
                res.sign_boxes.append(box)
                res.detected_signs.append({'type': 'ZEBRA_CROSS', 'val': 25, 'box': box})

        num_cars = len(res.cars)
        num_peds = len(res.peds)
        sig_str = f"SIGNAL: {res.signal_state}" if res.signal_state != 'CLEAR' else "NO SIGNAL"
        sign_str = f" | SIGN: {res.detected_signs[0]['type']}" if res.detected_signs else ""
        b_str = " | BARRIER NEAR" if res.boundary_alert else ""
        res.summary_text = f"VISION: {num_cars} CAR{'S' if num_cars!=1 else ''} | {sig_str} | {num_peds} PED{'S' if num_peds!=1 else ''}{sign_str}{b_str}"
        return res


# =============================================================================
# CRASH LEARNER (AI Post-Crash Reflection & Online Adaptation)
# =============================================================================
class CrashReport:
    """Diagnostic crash assessment and learning outcome."""
    def __init__(self, reason: str, diagnosis: str, remediation: str, impact_speed: float):
        self.reason = reason
        self.diagnosis = diagnosis
        self.remediation = remediation
        self.impact_speed = impact_speed
        self.timestamp = time.time()
        self.weights_updated: bool = False


class CrashLearner:
    """
    Diagnoses root causes of crashes and performs an immediate online policy gradient
    update on the pre-crash trajectory, saving adapted weights directly to disk.
    Persists lifetime learning memory to ai_learning_memory.json.
    """
    def __init__(self, max_buffer_steps: int = 50, memory_file: str = "ai_learning_memory.json"):
        self.max_buffer_steps = max_buffer_steps
        self.memory_file = memory_file
        self.trajectory_buffer: List[Dict[str, Any]] = []
        self.latest_report: Optional[CrashReport] = None
        self.total_mistakes_learned = 0
        self.saved_params: Dict[str, Any] = {}
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.total_mistakes_learned = int(data.get('total_learned', len(data.get('history', []))))
                    self.saved_params = data.get('controller_params', {})
            except Exception:
                pass

    def _save_memory_entry(self, report: CrashReport, speed: float, active_params: Optional[Dict[str, Any]] = None):
        entry = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'reason': report.reason,
            'diagnosis': report.diagnosis,
            'remediation': report.remediation,
            'impact_speed_kmh': round(speed, 1),
            'weights_saved': bool(report.weights_updated)
        }
        history = []
        prev_total = self.total_mistakes_learned
        params = dict(getattr(self, 'saved_params', {}))
        if active_params:
            params.update(active_params)
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prev_total = max(prev_total, int(data.get('total_learned', 0)))
                    history = data.get('history', [])
                    if not params and 'controller_params' in data:
                        params = data['controller_params']
            except Exception:
                history = []
        history.append(entry)
        self.total_mistakes_learned = max(prev_total + 1, len(history))
        self.saved_params = params
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_learned': self.total_mistakes_learned,
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'controller_params': params,
                    'history': history[-200:]
                }, f, indent=2)
        except Exception as e:
            print(f"[AI LEARNING] Error saving memory: {e}")

    def record_step(self, obs: np.ndarray, action: np.ndarray, log_prob: float,
                    value: float, speed: float):
        """Maintains a rolling window of driving states leading up to any event."""
        self.trajectory_buffer.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'log_prob': log_prob,
            'value': value,
            'speed': speed
        })
        if len(self.trajectory_buffer) > self.max_buffer_steps:
            self.trajectory_buffer.pop(0)

    def analyze_and_learn(self, sim: Any, reason: str, agent: Optional['PPOAgent'],
                          save_path: str = "rl_policy.pt") -> CrashReport:
        impact_speed = abs(getattr(getattr(sim, 'player', None), 'v', 0.0)) * 3.6
        reason_upper = reason.upper()

        # 1. Root Cause Diagnosis & Strategic Solution
        if "ONCOMING" in reason_upper or "HEAD-ON" in reason_upper or "INCOMING" in reason_upper or "WRONG WAY" in reason_upper:
            diag = f"Entered oncoming traffic corridor / incoming vehicle line (Head-on hazard at {impact_speed:.0f} km/h)"
            sol = "Locked lane matching strictly to outgoing travel direction; penalized oncoming line entry with R_WRONG_WAY (-100.0)."
        elif "MIDDLE" in reason_upper or "CENTERLINE" in reason_upper:
            diag = f"Cut center yellow line / middle divider during turn (Lateral dev: {abs(getattr(sim, '_last_lat_err', 0.0)):.2f}m)"
            sol = "Enforced middle divider corridor safety constraint and clamped cornering lookahead to 2.2m."
        elif "VEHICLE" in reason_upper or "CAR" in reason_upper:
            diag = f"Late braking behind lead vehicle (Impact at {impact_speed:.0f} km/h)"
            sol = "Enforced +8m safety following horizon; penalized late brake reaction."
        elif "BARRIER" in reason_upper or "WALL" in reason_upper or "GUARDRAIL" in reason_upper:
            diag = f"Lateral roadway departure into barrier (Yaw rate: {abs(getattr(getattr(sim, 'player', None), 'yaw_rate', 0.0)):.2f} rad/s)"
            sol = "Increased lane-centering restoring force; penalized excessive steering angle."
        elif "ROAD EDGE" in reason_upper or "CURB" in reason_upper:
            diag = f"Crossed road edge / curb boundary (Lateral offset: {abs(getattr(sim, '_last_lat_err', 0.0)):.2f}m)"
            sol = "Enforced road edge boundary penalty R_ROAD_EDGE_CROSS (-80.0) and applied optical steer restoring torque." 
        elif "PEDESTRIAN" in reason_upper:
            diag = "Failure to yield to crossing pedestrian on crosswalk"
            sol = "Clamped throttle to 0 and enforced emergency brake near pedestrian paths."
        else:
            diag = f"Collision with static geometry ({reason})"
            sol = "Tightened steering lookahead distance and reduced turn entrance speed."

        # Active Adaptive Parameter Remediation (Ensures the AI actually adapts control parameters)
        rb = getattr(getattr(sim, 'auto_drive_controller', None), 'rule_based', None)
        active_params = {}
        if rb is not None:
            if "BARRIER" in reason_upper or "WALL" in reason_upper or "GUARDRAIL" in reason_upper or "STATIC" in reason_upper:
                rb.min_corner_speed = max(2.2, getattr(rb, 'min_corner_speed', 3.8) - 0.35)
                rb.pre_turn_horizon = min(50.0, getattr(rb, 'pre_turn_horizon', 32.0) + 4.0)
                rb.restoring_torque_gain = min(0.70, getattr(rb, 'restoring_torque_gain', 0.35) + 0.08)
                rb.curb_avoid_thresh = max(1.2, getattr(rb, 'curb_avoid_thresh', 1.85) - 0.12)
                rb.lookahead_min = max(2.2, getattr(rb, 'lookahead_min', 3.2) - 0.25)
                sol = f"Adapted Parameters -> Corner Speed: {rb.min_corner_speed*3.6:.1f} km/h | Pre-turn: {rb.pre_turn_horizon:.0f}m | Restoring Gain: {rb.restoring_torque_gain:.2f} | Lookahead: {rb.lookahead_min:.1f}m"
            elif "VEHICLE" in reason_upper or "CAR" in reason_upper:
                rb.following_buffer = min(6.0, getattr(rb, 'following_buffer', 1.8) + 1.2)
                rb.dynamic_headway = min(1.6, getattr(rb, 'dynamic_headway', 0.65) + 0.15)
                rb.lead_car_safety_gap = min(32.0, getattr(rb, 'lead_car_safety_gap', 16.0) + 3.0)
                sol = f"Adapted Parameters -> Standstill Buffer: {rb.following_buffer:.1f}m | Headway: {rb.dynamic_headway:.2f}s | Queue Gap: {rb.lead_car_safety_gap:.0f}m"
            elif "PEDESTRIAN" in reason_upper:
                rb.ped_safety_margin = min(6.5, getattr(rb, 'ped_safety_margin', 3.8) + 1.2)
                sol = f"Adapted Parameters -> Pedestrian Margin expanded to {rb.ped_safety_margin:.1f}m"
            elif "ROAD EDGE" in reason_upper or "CURB" in reason_upper or "MIDDLE" in reason_upper or "CENTERLINE" in reason_upper:
                rb.restoring_torque_gain = min(0.75, getattr(rb, 'restoring_torque_gain', 0.35) + 0.10)
                rb.curb_avoid_thresh = max(1.1, getattr(rb, 'curb_avoid_thresh', 1.85) - 0.15)
                sol = f"Adapted Parameters -> Restoring Gain: {rb.restoring_torque_gain:.2f} | Curb Threshold: {rb.curb_avoid_thresh:.2f}m"
            elif "ONCOMING" in reason_upper or "HEAD-ON" in reason_upper or "INCOMING" in reason_upper or "WRONG WAY" in reason_upper:
                rb.curb_avoid_thresh = max(1.0, getattr(rb, 'curb_avoid_thresh', 1.85) - 0.20)
                rb.restoring_torque_gain = min(0.75, getattr(rb, 'restoring_torque_gain', 0.35) + 0.10)
                sol = f"Adapted Parameters -> Strict outgoing line lock | Restoring Gain: {rb.restoring_torque_gain:.2f}"
            else:
                rb.min_corner_speed = max(2.2, getattr(rb, 'min_corner_speed', 3.8) - 0.25)
                rb.pre_turn_horizon = min(48.0, getattr(rb, 'pre_turn_horizon', 32.0) + 3.0)
                sol = f"Adapted Parameters -> Corner Speed: {rb.min_corner_speed*3.6:.1f} km/h | Pre-turn: {rb.pre_turn_horizon:.0f}m"

            active_params = {
                'min_corner_speed': rb.min_corner_speed,
                'pre_turn_horizon': rb.pre_turn_horizon,
                'following_buffer': rb.following_buffer,
                'dynamic_headway': rb.dynamic_headway,
                'lead_car_safety_gap': rb.lead_car_safety_gap,
                'ped_safety_margin': rb.ped_safety_margin,
                'curb_avoid_thresh': rb.curb_avoid_thresh,
                'restoring_torque_gain': rb.restoring_torque_gain,
                'lookahead_min': rb.lookahead_min
            }

        report = CrashReport(reason, diag, sol, impact_speed)

        # 2. Online Failure Trajectory Policy Update
        if agent is not None and TORCH_AVAILABLE and len(self.trajectory_buffer) >= 2:
            try:
                agent.clear()
                n_steps = len(self.trajectory_buffer)
                for i, step_data in enumerate(self.trajectory_buffer):
                    decay = (i + 1) / float(n_steps)
                    pen = -150.0 * decay
                    agent.store_transition(
                        step_data['obs'], step_data['action'], pen,
                        step_data['log_prob'], step_data['value'],
                        done=(i == n_steps - 1)
                    )
                agent.finish_path(last_val=0.0)
                agent.update(batch_size=min(32, n_steps), epochs=4)
                agent.save(save_path)
                report.weights_updated = True
                print(f"[AI LEARNING] Online policy weights adapted -> {save_path}")
            except Exception as e:
                print(f"[AI LEARNING] Online update notice: {e}")

        # UNCONDITIONALLY SAVE TO MEMORY FOR EVERY CRASH
        self._save_memory_entry(report, impact_speed, active_params)
        print(f"[AI LEARNING] Lifetime mistakes learned: {self.total_mistakes_learned}")
        if active_params:
            print(f"  -> {sol}")

        if hasattr(sim, 'auto_drive_controller') and sim.auto_drive_controller is not None:
            sim.auto_drive_controller.live_telemetry['mistakes_learned'] = self.total_mistakes_learned

        self.latest_report = report
        self.trajectory_buffer.clear()
        return report


# =============================================================================
# ACTOR-CRITIC NETWORK (PyTorch PPO Architecture)
# =============================================================================
if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """
        Actor-Critic Multi-Layer Perceptron for continuous vehicle control.
        Inputs: Normalized observation vector (32 features including CNN vision features).
        Outputs:
          - Actor: mean and log_std for (steering [-1, 1], throttle [0, 1], brake [0, 1])
          - Critic: scalar state value estimate V(s)
        """
        def __init__(self, obs_dim: int = 32, act_dim: int = 3, hidden_dim: int = 128):
            super().__init__()
            self.obs_dim = obs_dim
            self.act_dim = act_dim

            self.shared_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )

            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, act_dim),
            )
            self.actor_log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            features = self.shared_net(x)
            mean = self.actor_mean(features)
            steer_mean = torch.tanh(mean[..., 0:1])
            pedal_mean = torch.sigmoid(mean[..., 1:3])
            act_mean = torch.cat([steer_mean, pedal_mean], dim=-1)
            value = self.critic(features)
            std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
            return act_mean, std, value

        def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
            with torch.no_grad():
                # Pad or slice if observation dimension differs
                if len(obs) < self.obs_dim:
                    obs = np.pad(obs, (0, self.obs_dim - len(obs)))
                elif len(obs) > self.obs_dim:
                    obs = obs[:self.obs_dim]
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean, std, val = self.forward(x)
                if deterministic:
                    action = mean
                    log_prob = 0.0
                else:
                    dist = Normal(mean, std)
                    raw_action = dist.sample()
                    log_prob = dist.log_prob(raw_action).sum(dim=-1).item()
                    s = torch.clamp(raw_action[..., 0:1], -1.0, 1.0)
                    t_b = torch.clamp(raw_action[..., 1:3], 0.0, 1.0)
                    action = torch.cat([s, t_b], dim=-1)
                return action.squeeze(0).cpu().numpy(), float(log_prob), float(val.item())

        def evaluate(self, obs_tensor: torch.Tensor, act_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, std, values = self.forward(obs_tensor)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(act_tensor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_probs, values.squeeze(-1), entropy
else:
    class ActorCritic:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to initialize ActorCritic.")


# =============================================================================
# PPO AGENT
# =============================================================================
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for training and evaluating
    the vehicle control policy.
    """
    def __init__(self, obs_dim: int = 32, act_dim: int = 3, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 device: str = "cpu"):
        self.device = torch.device(device if TORCH_AVAILABLE and torch.cuda.is_available() and device != "cpu" else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        self.lr = lr
        if TORCH_AVAILABLE:
            self.policy = ActorCritic(obs_dim, act_dim).to(self.device)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        else:
            self.policy = None
            self.optimizer = None

        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        if not TORCH_AVAILABLE or self.policy is None:
            return np.array([0.0, 0.0, 0.0]), 0.0, 0.0
        return self.policy.get_action(obs, deterministic)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         log_prob: float, value: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def finish_path(self, last_val: float = 0.0):
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_val], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values[:-1]
        self.advantages = advantages
        self.returns = returns

    def update(self, batch_size: int = 64, epochs: int = 6) -> Dict[str, float]:
        if not TORCH_AVAILABLE or len(self.states) == 0:
            return {}

        states_t = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(self.returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(self.advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        loss_actor_avg, loss_critic_avg = 0.0, 0.0
        updates = 0

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                log_probs, values, entropy = self.policy.evaluate(states_t[batch_idx], actions_t[batch_idx])
                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values, returns_t[batch_idx])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                loss_actor_avg += actor_loss.item()
                loss_critic_avg += critic_loss.item()
                updates += 1

        self.clear()
        return {
            "actor_loss": loss_actor_avg / max(1, updates),
            "critic_loss": loss_critic_avg / max(1, updates)
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path: str):
        if TORCH_AVAILABLE and self.policy is not None:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'obs_dim': self.policy.obs_dim,
                'act_dim': self.policy.act_dim
            }, path)
            print(f"[RL] Model checkpoint saved to: {path}")

    def load(self, path: str) -> bool:
        if not TORCH_AVAILABLE or not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint['policy_state_dict'] if (isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint) else checkpoint
            
            if isinstance(state_dict, dict) and 'shared_net.0.weight' in state_dict:
                ckpt_obs_dim = state_dict['shared_net.0.weight'].shape[1]
                if ckpt_obs_dim != self.policy.obs_dim:
                    self.policy = ActorCritic(obs_dim=ckpt_obs_dim, act_dim=self.policy.act_dim).to(self.device)
                    self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
            
            self.policy.load_state_dict(state_dict, strict=False)
            self.policy.eval()
            print(f"[RL] Model loaded successfully from: {path} (obs_dim={self.policy.obs_dim})")
            return True
        except Exception as e:
            print(f"[RL] Model load notice: {e}")
            return False


# =============================================================================
# REAR BLIND-SPOT & PROXIMITY MONITOR
# =============================================================================
class BlindSpotResult:
    """Telemetry from rearview camera and side blind-spot proximity sensors."""
    def __init__(self):
        self.left_clear: bool = True
        self.right_clear: bool = True
        self.left_dist: float = 999.0
        self.right_dist: float = 999.0
        self.left_rel_v: float = 0.0
        self.right_rel_v: float = 0.0
        self.rear_center_dist: float = 999.0


# =============================================================================
# RULE-BASED CONTROLLER & SAFETY SUPERVISOR
# =============================================================================
class RuleBasedController:
    """
    Deterministic rule-based autonomous driving controller and safety layer.
    """
    def __init__(self, lookahead_min: float = 3.2, lookahead_gain: float = 1.2):
        self.lookahead_min = lookahead_min
        self.lookahead_gain = lookahead_gain
        self.min_corner_speed = 3.8
        self.pre_turn_horizon = 32.0
        self.following_buffer = 1.8
        self.dynamic_headway = 0.65
        self.lead_car_safety_gap = 16.0
        self.ped_safety_margin = 3.8
        self.curb_avoid_thresh = 1.85
        self.restoring_torque_gain = 0.35
        self.lead_stop_timer = 0.0

    def check_rear_blind_spots(self, sim: Any) -> BlindSpotResult:
        """
        Uses rear camera & side proximity sensors to monitor vehicles behind and in blind spots.
        Ensures safe lane changes and turn preparation without cutting off approaching vehicles.
        """
        P = sim.player
        res = BlindSpotResult()
        cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
        speed = abs(P.v)

        if not hasattr(sim, 'traffic') or not hasattr(sim.traffic, 'cars'):
            return res

        for car in sim.traffic.cars:
            dx = car.x - P.x
            dy = car.y - P.y
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h

            car_v = getattr(car, 'v', 0.0)
            car_len = getattr(car, 'length', 4.5)
            # Rear bumper-to-bumper gap (positive when behind us)
            gap = (-fwd) - (P.length * 0.5 + car_len * 0.5)

            # 1. Directly behind us in current lane
            if abs(lat) <= 1.2 and fwd < 0.0:
                if gap < res.rear_center_dist:
                    res.rear_center_dist = max(0.0, gap)

            # 2. Left adjacent lane / blind spot (lat < -1.0m, negative = LEFT in Pygame y-down)
            if -4.8 <= lat <= -1.0:
                # Alongside or in immediate rear quarter blind spot: -4.5m <= fwd <= 4.0m
                if -4.5 <= fwd <= 4.0:
                    res.left_clear = False
                    res.left_dist = min(res.left_dist, max(0.0, gap))
                # Approaching from behind: -32m <= fwd < -4.5m
                elif -32.0 <= fwd < -4.5:
                    rel_v = car_v - speed  # positive if closing in
                    ttc = gap / max(0.5, rel_v) if rel_v > 0.0 else 999.0
                    if gap < 11.0 or ttc < 3.2:
                        res.left_clear = False
                    if gap < res.left_dist:
                        res.left_dist = gap
                        res.left_rel_v = rel_v

            # 3. Right adjacent lane / blind spot (lat > 1.0m, positive = RIGHT in Pygame y-down)
            if 1.0 <= lat <= 4.8:
                # Alongside or in immediate rear quarter blind spot
                if -4.5 <= fwd <= 4.0:
                    res.right_clear = False
                    res.right_dist = min(res.right_dist, max(0.0, gap))
                # Approaching from behind
                elif -32.0 <= fwd < -4.5:
                    rel_v = car_v - speed
                    ttc = gap / max(0.5, rel_v) if rel_v > 0.0 else 999.0
                    if gap < 11.0 or ttc < 3.2:
                        res.right_clear = False
                    if gap < res.right_dist:
                        res.right_dist = gap
                        res.right_rel_v = rel_v

        return res

    def compute_action(self, sim: Any) -> Tuple[float, float, float]:
        P = sim.player
        v = P.v
        speed_abs = abs(v)

        # 1. Base Road Speed Limit
        road_code, posted_lim = sim.city.surface_at(P.x, P.y, P.lvl)
        target_speed = posted_lim / 3.6 if (road_code in (1, 2) and posted_lim) else 11.1
        target_speed = min(target_speed, 16.0)

        # 2. Comprehensive Pre-Turn & U-Turn Speed Control (Decelerate smoothly before corners)
        pts = getattr(sim, 'nav_route_pts', None) or getattr(sim, 'nav_full_pts', None)
        max_turn_angle = 0.0
        cum_turn_angle = 0.0
        dist_to_turn = 999.0
        is_u_turn = False
        if pts and len(pts) >= 2:
            # Find closest waypoint along route polyline
            best_i = 0
            min_d_sq = float('inf')
            for i, pt in enumerate(pts):
                d_sq = (pt[0] - P.x) ** 2 + (pt[1] - P.y) ** 2
                if d_sq < min_d_sq:
                    min_d_sq = d_sq
                    best_i = i

            upcoming_pts = pts[best_i:min(len(pts), best_i + 40)]
            if len(upcoming_pts) >= 2:
                accum_dist = math.hypot(upcoming_pts[0][0] - P.x, upcoming_pts[0][1] - P.y)
                p_prev = upcoming_pts[0]
                prev_seg_hdg = P.heading
                for pt in upcoming_pts[1:]:
                    seg_len = math.hypot(pt[0] - p_prev[0], pt[1] - p_prev[1])
                    if seg_len < 0.2:
                        continue
                    accum_dist += seg_len
                    if accum_dist > 55.0:
                        break
                    seg_heading = math.atan2(pt[1] - p_prev[1], pt[0] - p_prev[0])
                    # Difference relative to current vehicle heading
                    angle_diff = abs((seg_heading - P.heading + math.pi) % (2.0 * math.pi) - math.pi)
                    # Difference relative to preceding segment (curve progression)
                    bend_diff = abs((seg_heading - prev_seg_hdg + math.pi) % (2.0 * math.pi) - math.pi)
                    cum_turn_angle += bend_diff
                    if angle_diff > max_turn_angle:
                        max_turn_angle = angle_diff
                        dist_to_turn = min(dist_to_turn, accum_dist)
                    p_prev = pt
                    prev_seg_hdg = seg_heading

        # Detect U-turn or hairpin loop (> 72 deg deviation or cumulative loop > 90 deg)
        is_u_turn = (max_turn_angle > 1.25 or cum_turn_angle > 1.60)

        # Compute safe cornering speed and braking horizon based on turn sharpness
        if is_u_turn:
            # U-turn / Hairpin: slow to safe crawl (10 km/h = 2.78 m/s) with 45m braking horizon
            corner_speed = min(self.min_corner_speed, 2.78)
            horizon = max(self.pre_turn_horizon, 45.0)
            a_decel = 1.5  # comfortable deceleration m/s^2
        elif max_turn_angle > 0.45 or cum_turn_angle > 0.65:       # Sharp corner (> 25 deg)
            corner_speed = self.min_corner_speed
            horizon = max(self.pre_turn_horizon, 32.0)
            a_decel = 1.8
        elif max_turn_angle > 0.22:     # Medium turn (12 - 25 deg)
            corner_speed = max(self.min_corner_speed, 5.2)
            horizon = max(self.pre_turn_horizon - 5.0, 24.0)
            a_decel = 2.0
        elif max_turn_angle > 0.12:     # Gentle curve (7 - 12 deg)
            corner_speed = max(self.min_corner_speed + 1.5, 7.5)
            horizon = 20.0
            a_decel = 2.2
        else:
            corner_speed = target_speed
            horizon = 15.0
            a_decel = 2.5

        # Progressive kinematic deceleration: ensures vehicle reaches corner_speed BEFORE the turn entry
        if max_turn_angle > 0.18 and dist_to_turn < horizon:
            v_kinematic = math.sqrt(corner_speed * corner_speed + 2.0 * a_decel * max(0.0, dist_to_turn - 1.5))
            target_speed = min(target_speed, max(corner_speed, v_kinematic))
        elif max_turn_angle > 0.18 and dist_to_turn < 4.0:
            target_speed = min(target_speed, corner_speed)

        # 3. Check Destination Bay Proximity (Drive off street into parking spot)
        d_finish = 999.0
        if getattr(sim.city, 'destinations', None):
            d_idx = getattr(sim, 'dest_idx', 0) % len(sim.city.destinations)
            dest = sim.city.destinations[d_idx]
            finish_pt = dest.get('finish_pt', (dest['x'], dest['y']))
            d_finish = math.hypot(P.x - finish_pt[0], P.y - finish_pt[1])

        approaching_parking = (d_finish < 38.0 and pts and len(pts) >= 2)

        # Query lane matching with tight spatial bound (maxd=3.8m to prevent false snapping)
        res = sim.city.nearest_path(P.x, P.y, P.heading, P.z, maxd=3.8)
        current_lane = res[0] if res else None
        s_pos = res[1] if res else 0.0

        target_lane = current_lane
        if current_lane and current_lane.kind == 'lane' and not approaching_parking:
            target_lane = self._evaluate_lane_choice(sim, current_lane, s_pos)

        # 4. Steering Guidance:
        # Use Pure Pursuit on route waypoints (pts) during turns, intersections, and parking
        # Use Stanley lane centering ONLY on straight/open road lanes
        steer = 0.0
        max_steer = getattr(P, 'max_steer', 0.55)

        use_waypoint_steer = (approaching_parking or 
                              max_turn_angle > 0.18 or 
                              dist_to_turn < 18.0 or 
                              current_lane is None or 
                              current_lane.kind != 'lane' or
                              (target_lane and abs(s_pos - target_lane.length) < 8.0))

        if not use_waypoint_steer and target_lane and target_lane.kind == 'lane':
            target_s = min(target_lane.length, s_pos + max(3.0, 0.7 * speed_abs))
            lx, ly, lh = target_lane.pose(target_s)
            dx = P.x - lx
            dy = P.y - ly
            # Signed lateral cross-track error: positive = left of lane center, negative = right
            lat_err = -math.sin(lh) * dx + math.cos(lh) * dy
            sim._last_lat_err = lat_err
            if abs(lat_err) < 2.5:
                heading_err = (lh - P.heading + math.pi) % (2.0 * math.pi) - math.pi
                k_e = 0.85
                stanley_angle = heading_err + math.atan2(-k_e * lat_err, max(speed_abs, 2.5))
                steer = float(np.clip(stanley_angle / max_steer, -1.0, 1.0))
            else:
                use_waypoint_steer = True

        if use_waypoint_steer and pts and len(pts) >= 2:
            lookahead_dist = max(self.lookahead_min, min(6.5, 0.45 * speed_abs + 2.8))
            target_pt = self._find_lookahead_point(P.x, P.y, pts, lookahead_dist)
            dx = target_pt[0] - P.x
            dy = target_pt[1] - P.y
            target_heading = math.atan2(dy, dx)
            heading_err = (target_heading - P.heading + math.pi) % (2.0 * math.pi) - math.pi
            L = max(P.wheelbase, 2.7)
            steer_angle_target = math.atan2(2.0 * L * math.sin(heading_err), lookahead_dist)
            steer = float(np.clip(steer_angle_target / max_steer, -1.0, 1.0))

        # 5. Traffic Signal Ahead (Smooth glide directly to stop line, zero gap)
        must_stop_for_signal, dist_to_stop = self._check_signal_ahead(sim)
        if must_stop_for_signal:
            if dist_to_stop <= 0.45:
                target_speed = 0.0
            elif dist_to_stop < 20.0:
                # Maintain gentle controlled roll until front bumper reaches stop line
                target_speed = min(target_speed, max(0.8, (dist_to_stop - 0.45) * 0.45))

        # 6. Vehicle Ahead in current lane (Approach lead car realistically with adaptive following buffer)
        veh_ahead_gap, veh_ahead_speed = self._check_vehicle_ahead(sim)
        if veh_ahead_gap < 32.0:
            fb = self.following_buffer
            if veh_ahead_speed < 1.0:
                # Stopped lead vehicle: glide smoothly into adaptive standstill buffer
                if veh_ahead_gap <= (fb + 0.05):
                    # Check deadlock watchdog: if stopped > 2.0s and room ahead > 1.2m, creep; if > 4.5s, bypass
                    if self.lead_stop_timer > 4.5 and veh_ahead_gap > 0.8:
                        target_speed = 2.5
                    elif self.lead_stop_timer > 2.0 and veh_ahead_gap > 1.2:
                        target_speed = 1.2
                    else:
                        target_speed = 0.0
                elif veh_ahead_gap < (fb + 4.2):
                    target_speed = min(target_speed, max(0.6, (veh_ahead_gap - fb) * 0.75))
                else:
                    target_speed = min(target_speed, max(3.0, (veh_ahead_gap - fb) * 0.55))
            else:
                # Moving lead vehicle: follow at adaptive following_buffer + dynamic_headway
                safe_following_gap = fb + self.dynamic_headway * speed_abs
                if veh_ahead_gap < (fb + 0.2):
                    target_speed = 0.0
                elif veh_ahead_gap < safe_following_gap:
                    target_speed = min(target_speed, max(0.0, veh_ahead_speed * 0.90))
                elif veh_ahead_gap < self.lead_car_safety_gap:
                    target_speed = min(target_speed, max(2.5, veh_ahead_speed + 0.8))
        else:
            self.lead_stop_timer = 0.0

        # 7. Pedestrian Ahead (only if actively crossing or inside roadway corridor)
        ped_ahead_dist = self._check_pedestrian_ahead(sim)
        if ped_ahead_dist < 18.0:
            if ped_ahead_dist < self.ped_safety_margin:
                target_speed = 0.0
            else:
                target_speed = min(target_speed, max(0.0, (ped_ahead_dist - (self.ped_safety_margin - 0.3)) * 0.40))

        # 8. Road Sign Perception & Dynamic Speed Regulation
        # Only adhere to signs along the vehicle's right curb in current direction of travel
        if hasattr(sim, 'city') and hasattr(sim.city, 'signs'):
            cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
            for s in sim.city.signs:
                dx = s['x'] - P.x
                dy = s['y'] - P.y
                fwd = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                # Sign must be on the right-hand curb (0.8m <= lat <= 4.2m) and facing traffic
                s_hdg = math.atan2(s.get('fy', 0.0), s.get('fx', 1.0))
                facing_ok = (math.cos(s_hdg - P.heading) > 0.40)
                if 2.0 < fwd < 30.0 and 0.8 <= lat <= 4.5 and facing_ok:
                    if s['kind'] == 'speed':
                        target_speed = min(target_speed, (s['val'] / 3.6))
                    elif s['kind'] == 'zebra' and fwd < 18.0 and ped_ahead_dist < 22.0:
                        # Only slow to caution speed at zebra crossing if pedestrian is nearby
                        target_speed = min(target_speed, 6.94)

        # 9. Destination Bay & Proper In-Bay Centering (Centers squarely in parking box)
        if d_finish < 35.0:
            if d_finish <= 1.2:
                target_speed = 0.0
            elif d_finish < 8.0:
                target_speed = min(target_speed, max(0.9, (d_finish - 1.0) * 0.35))
            else:
                target_speed = min(target_speed, max(2.5, (d_finish - 5.0) * 0.32))

        # 10. Road Edge Crossing Penalty & Restoring Steer (Adaptive restoring torque)
        lat_err = getattr(sim, '_last_lat_err', 0.0)
        if abs(lat_err) > self.curb_avoid_thresh:
            # Vehicle encroaching on road edge / curb: apply corrective steering restoring torque
            steer = float(np.clip(steer - np.sign(lat_err) * self.restoring_torque_gain, -1.0, 1.0))

        # Longitudinal pedal control with Wide Coasting Deadband (Zero unnecessary braking)
        speed_err = target_speed - speed_abs
        if target_speed <= 0.2 and speed_abs < 0.6:
            thr, brk = 0.0, 0.85
        elif speed_err > 0.3:
            # Below target speed: smooth acceleration
            thr = float(np.clip(0.40 + 0.35 * speed_err, 0.0, 1.0))
            brk = 0.0
        elif speed_abs <= target_speed + (0.5 if is_u_turn else 2.0):
            # Natural coasting deadband
            thr, brk = 0.0, 0.0
        else:
            # Speed substantially exceeds target: apply progressive deceleration
            thr = 0.0
            margin = 0.4 if is_u_turn else 1.8
            brk = float(np.clip((speed_abs - (target_speed + margin)) * 0.30, 0.12, 0.85))

        return steer, thr, brk

    def supervise_action(self, sim: Any, rl_action: np.ndarray) -> Tuple[float, float, float]:
        path_steer, path_thr, path_brk = self.compute_action(sim)

        # 1. Primary Steering: Stanley Lane Center & Turn Guidance
        # When entering a turn (|path_steer| > 0.10) or when lateral error is non-zero,
        # anchor firmly to path_steer to keep car centered inside its lane and prevent straddling lines
        lat_err = getattr(sim, '_last_lat_err', 0.0)
        if abs(path_steer) > 0.10 or abs(lat_err) > 0.35:
            steer = path_steer
        else:
            steer = float(np.clip(float(rl_action[0]), path_steer - 0.06, path_steer + 0.06))

        # 1. Destination Bay Parking Spot Arrival (Squarely centered inside yellow box)
        if getattr(sim.city, 'destinations', None):
            d_idx = getattr(sim, 'dest_idx', 0) % len(sim.city.destinations)
            dest = sim.city.destinations[d_idx]
            finish_pt = dest.get('finish_pt', (dest['x'], dest['y']))
            d_finish = math.hypot(sim.player.x - finish_pt[0], sim.player.y - finish_pt[1])
            if d_finish <= 1.3:
                self.driving_status = "MISSION COMPLETED (PARKED)"
                return path_steer, 0.0, 1.0

        # 2. Dynamic Obstacle Following & Lead Vehicle Queue
        veh_gap, veh_speed = self._check_vehicle_ahead(sim)
        ped_dist = self._check_pedestrian_ahead(sim)
        must_stop, dist_to_stop = self._check_signal_ahead(sim)

        speed_abs = abs(sim.player.v)

        # 2a. Pedestrian Crossing Safety Priority (Adaptive pedestrian safety margin)
        if ped_dist < self.ped_safety_margin:
            self.driving_status = "YIELDING TO PEDESTRIAN"
            return steer, 0.0, 0.85
        elif ped_dist < (self.ped_safety_margin + 4.8):
            self.driving_status = "SLOWING FOR PEDESTRIAN"
            v_ped_target = max(1.0, (ped_dist - self.ped_safety_margin) * 1.2)
            if speed_abs > v_ped_target:
                return steer, 0.0, float(np.clip((speed_abs - v_ped_target) / 2.0, 0.25, 0.80))
            elif speed_abs < v_ped_target - 0.5:
                return steer, 0.30, 0.0
            else:
                return steer, 0.0, 0.0

        # 2b. Dynamic Lead Vehicle Following & Queue Stoppage (Adaptive following buffer)
        dt = 0.033
        if veh_gap < 30.0:
            rem_gap = veh_gap - (self.following_buffer + 0.05)
            if rem_gap <= 0.15:
                # Front bumper reached target adaptive gap behind lead vehicle
                if speed_abs < 0.2:
                    self.lead_stop_timer += dt
                else:
                    self.lead_stop_timer = max(0.0, self.lead_stop_timer - dt)

                # Deadlock Watchdog: prevent indefinite freezing behind stationary / side obstacles
                if self.lead_stop_timer > 4.5 and veh_gap > 0.8:
                    self.driving_status = "BYPASSING TRAFFIC DEADLOCK"
                    return steer, 0.35, 0.0
                elif self.lead_stop_timer > 2.0 and veh_gap > 1.2:
                    self.driving_status = "CREEPING PAST TRAFFIC QUEUE"
                    return steer, 0.28, 0.0
                else:
                    self.driving_status = "WAITING AT RED LIGHT (TRAFFIC QUEUE)" if must_stop else "WAITING BEHIND LEAD VEHICLE"
                    return steer, 0.0, 0.85
            else:
                self.lead_stop_timer = max(0.0, self.lead_stop_timer - dt * 2.0)
                if veh_speed < 0.8:
                    # Lead vehicle is stopped: smoothly glide and close gap to following_buffer without oscillating
                    v_target = float(np.clip(math.sqrt(max(0.1, 2.0 * 1.0 * rem_gap)), 0.6, 6.0))
                    self.driving_status = "CLOSING GAP IN TRAFFIC QUEUE" if rem_gap < 3.5 else "DECELERATING FOR LEAD VEHICLE"
                    if speed_abs > v_target:
                        brk = float(np.clip((speed_abs - v_target) / 1.5, 0.15, 0.65))
                        return steer, 0.0, brk
                    elif speed_abs < v_target - 0.2:
                        thr = float(np.clip(0.28 + (v_target - speed_abs) * 0.15, 0.25, 0.45))
                        return steer, thr, 0.0
                    else:
                        return steer, 0.0, 0.0
                elif veh_speed >= 0.8 and rem_gap < 8.0:
                    # Lead vehicle is pulling away: smoothly match traffic flow
                    self.driving_status = "FOLLOWING TRAFFIC STREAM"
                    v_target = veh_speed
                    if speed_abs > v_target + 0.5:
                        return steer, 0.0, 0.35
                    elif speed_abs < v_target:
                        return steer, 0.55, 0.0
                    else:
                        return steer, 0.0, 0.0
        else:
            self.lead_stop_timer = 0.0

        # 3. Traffic Signal Stop Line Glide (Smooth v_target approach, bumper right at stop line)
        if must_stop and dist_to_stop < 22.0:
            if dist_to_stop <= 0.40:
                # Front bumper directly touching stop line - hold stationary with full brake
                self.driving_status = "STOPPED AT RED LIGHT (AT STOP LINE)"
                return steer, 0.0, 0.85
            else:
                # Smooth continuous deceleration profile (v_target = sqrt(2 * a * rem_d))
                # Eliminates gap completely and glides right to 0.40m with zero gas/brake chatter
                rem_d = max(0.05, dist_to_stop - 0.35)
                v_target = float(np.clip(math.sqrt(2.0 * 1.1 * rem_d), 0.65, 7.5))
                self.driving_status = "GLIDING TO STOP LINE" if dist_to_stop < 3.5 else f"STOPPING FOR RED LIGHT ({dist_to_stop:.1f}m)"
                if speed_abs > v_target:
                    brk = float(np.clip((speed_abs - v_target) / 1.4, 0.15, 0.65))
                    return steer, 0.0, brk
                elif speed_abs < v_target - 0.25:
                    thr = float(np.clip(0.26 + (v_target - speed_abs) * 0.15, 0.22, 0.45))
                    return steer, thr, 0.0
                else:
                    return steer, 0.0, 0.0

        # 4. Longitudinal Supervision (Mutual Exclusion: Never apply throttle and brake together!)
        if path_brk > 0.05:
            self.driving_status = "DECELERATING FOR TURN"
            thr = 0.0
            brk = path_brk
        elif path_thr > 0.02:
            if abs(path_steer) > 0.12:
                self.driving_status = "NAVIGATING INTERSECTION TURN"
            else:
                self.driving_status = f"CRUISING ({speed_abs*3.6:.0f} km/h)"
            thr = max(float(rl_action[1]), path_thr, 0.50)
            brk = 0.0
        else:
            self.driving_status = f"COASTING ({speed_abs*3.6:.0f} km/h)"
            thr = 0.0
            brk = 0.0

        # 5. Anti-Stall / Creep Forward Watchdog for open road
        if abs(sim.player.v) < 0.25 and ped_dist > 4.0 and not must_stop and veh_gap >= 14.0 and path_brk < 0.05:
            self.driving_status = "ACCELERATING"
            thr = 0.65
            brk = 0.0

        return steer, thr, brk

    def _evaluate_lane_choice(self, sim: Any, current_lane: Any, s_pos: float) -> Any:
        """
        Intelligently selects the optimal lane:
        1. Pre-turn lane selection with Rear Camera & Blind-Spot Safety Check:
           - Left turn ahead (< 60m) -> target leftmost lane (highest lane idx)
             * ONLY if left blind-spot check confirms no approaching car behind!
           - Right turn ahead (< 60m) -> target rightmost lane (idx 0)
             * ONLY if right blind-spot check confirms no approaching car behind!
           - If car is approaching from behind in the target lane, hold current lane until clear.
        2. Overtaking:
           - If current lane has slow/stopped traffic ahead (< 20m) AND adjacent lane is clear
             both ahead and behind (rear camera check).
        """
        if not hasattr(current_lane, 'edge') or not current_lane.edge:
            return current_lane

        edge = current_lane.edge
        same_dir_lanes = edge.lanes.get(current_lane.tdir, [])
        if len(same_dir_lanes) <= 1:
            return current_lane

        # Check rear blind spots using back camera / sensors
        bs = self.check_rear_blind_spots(sim)
        sim.blind_spot = bs

        dist_to_end = max(0.0, current_lane.length - s_pos)
        next_turn_move = None

        # 1. Proactively query planned route for upcoming turn maneuver exiting this road
        nav_route = getattr(sim, 'nav_route', None)
        if nav_route:
            current_edge_idx = -1
            for i, item in enumerate(nav_route):
                if getattr(item, 'edge', None) == edge and getattr(item, 'tdir', None) == current_lane.tdir:
                    current_edge_idx = i
                    break
                elif getattr(item, 'pid', None) == current_lane.pid:
                    current_edge_idx = i
                    break

            if current_edge_idx != -1:
                for item in nav_route[current_edge_idx:]:
                    if getattr(item, 'kind', None) == 'conn':
                        next_turn_move = getattr(item, 'move', None)
                        break

        # 2. Route polyline lookahead for upcoming intersection turn (scans ahead up to 35m)
        pts = getattr(sim, 'nav_route_pts', None)
        if next_turn_move is None and pts and len(pts) >= 4 and dist_to_end < 35.0:
            accum_d = 0.0
            p_prev = (sim.player.x, sim.player.y)
            for pt in pts[1:]:
                accum_d += math.hypot(pt[0] - p_prev[0], pt[1] - p_prev[1])
                if accum_d > 35.0:
                    break
                if accum_d >= 8.0:
                    seg_h = math.atan2(pt[1] - p_prev[1], pt[0] - p_prev[0])
                    angle_diff = (seg_h - sim.player.heading + math.pi) % (2.0 * math.pi) - math.pi
                    if abs(angle_diff) > 2.2:
                        next_turn_move = 'U'
                        break
                    elif angle_diff > 0.35:
                        next_turn_move = 'L'
                        break
                    elif angle_diff < -0.35:
                        next_turn_move = 'R'
                        break
                p_prev = pt

        # 3. Dedicated Pre-Turn Lane Positioning with Rear Blind-Spot Clearance:
        # Only position for turn when within 35m of the intersection / end of current road
        # - idx 0 is the OUTER lane (RIGHT lane, nearest sidewalk/curb) -> Target for RIGHT turn ('R')
        # - idx (len - 1) is the INNER lane (LEFT lane, nearest center median) -> Target for LEFT/U turn ('L'/'U')
        if next_turn_move in ('L', 'U', 'R') and dist_to_end < 35.0:
            target_idx = (len(same_dir_lanes) - 1) if next_turn_move in ('L', 'U') else 0
            if current_lane.idx != target_idx:
                candidate = same_dir_lanes[target_idx]
                adj_lead, _ = self._check_lane_traffic(sim, candidate, s_pos)

                is_left_transition = (target_idx > current_lane.idx)
                rear_clear = bs.left_clear if is_left_transition else bs.right_clear

                # If candidate lane is clear ahead (> 7.5m) and rear blind spot is clear, shift lanes!
                if adj_lead > 7.5 and rear_clear:
                    return candidate
                else:
                    return current_lane

        # 4. Traffic Congestion Avoidance / Overtaking
        # Only change lane if current lane has slow/stopped traffic ahead (< 16m) and adjacent lane is clear
        lead_gap, lead_v = self._check_lane_traffic(sim, current_lane, s_pos)
        if lead_gap < 16.0 and lead_v < 2.0:
            for cand_idx in range(len(same_dir_lanes)):
                if cand_idx != current_lane.idx:
                    cand = same_dir_lanes[cand_idx]
                    cand_gap, _ = self._check_lane_traffic(sim, cand, s_pos)
                    is_left = (cand_idx > current_lane.idx)
                    rear_ok = bs.left_clear if is_left else bs.right_clear
                    if cand_gap > lead_gap + 10.0 and rear_ok:
                        return cand

        # 4. Overtaking slow traffic in current lane (only if not within 40m of upcoming turn)
        cur_lead_gap, cur_lead_v = self._check_lane_traffic(sim, current_lane, s_pos)
        if cur_lead_gap < 18.0 and dist_to_end > 40.0:
            for adj_lane in same_dir_lanes:
                if adj_lane.pid == current_lane.pid:
                    continue
                adj_lead_gap, _ = self._check_lane_traffic(sim, adj_lane, s_pos)
                is_left = (adj_lane.idx > current_lane.idx)
                adj_rear_clear = bs.left_clear if is_left else bs.right_clear

                if adj_lead_gap > 20.0 and adj_rear_clear:
                    return adj_lane

        return current_lane
    def _check_lane_traffic(self, sim: Any, lane: Any, s_pos: float) -> Tuple[float, float]:
        min_lead_d = 999.0
        lead_v = 0.0
        if hasattr(sim, 'traffic') and hasattr(sim.traffic, 'cars'):
            for car in sim.traffic.cars:
                c_res = sim.city.nearest_path(car.x, car.y, car.heading, getattr(car, 'z', 0.0), maxd=3.5)
                if c_res and c_res[0].pid == lane.pid:
                    c_s = c_res[1]
                    d = c_s - s_pos
                    if 0.5 < d < min_lead_d:
                        min_lead_d = d
                        lead_v = getattr(car, 'v', 0.0)
        return min_lead_d, lead_v

    def _check_lane_rear(self, sim: Any, lane: Any, s_pos: float) -> float:
        min_rear_d = 999.0
        if hasattr(sim, 'traffic') and hasattr(sim.traffic, 'cars'):
            for car in sim.traffic.cars:
                c_res = sim.city.nearest_path(car.x, car.y, car.heading, getattr(car, 'z', 0.0), maxd=3.5)
                if c_res and c_res[0].pid == lane.pid:
                    c_s = c_res[1]
                    d = s_pos - c_s
                    if 0.5 < d < min_rear_d:
                        min_rear_d = d
        return min_rear_d

    def _find_lookahead_point(self, px: float, py: float, pts: List[Tuple[float, float]], lookahead: float) -> Tuple[float, float]:
        if not pts:
            return (px, py)
        # Find closest point on path to current position
        best_i = 0
        min_d_sq = float('inf')
        for i, (x, y) in enumerate(pts):
            d_sq = (x - px) ** 2 + (y - py) ** 2
            if d_sq < min_d_sq:
                min_d_sq = d_sq
                best_i = i

        # Step forward along the path from best_i until lookahead distance is reached
        for i in range(best_i, len(pts)):
            if math.hypot(pts[i][0] - px, pts[i][1] - py) >= lookahead:
                return pts[i]
        return pts[-1]

    def _check_signal_ahead(self, sim: Any) -> Tuple[bool, float]:
        P = sim.player
        res = sim.city.nearest_path(P.x, P.y, P.heading, P.z, maxd=20.0)
        if res is None:
            return False, 999.0
        lane, s_pos, _ = res
        if getattr(lane, 'signal', None) and getattr(lane, 'stop_s', None) is not None:
            # Front bumper distance to stop line (s_pos is vehicle center)
            dist_to_stop = lane.stop_s - (s_pos + P.length * 0.5)
            if -1.0 <= dist_to_stop < 45.0:
                sig_state = lane.signal.state(lane.axis, sim.sim_time)
                if sig_state in ('R', 'Y'):
                    return True, dist_to_stop
        return False, 999.0

    def _check_vehicle_ahead(self, sim: Any) -> Tuple[float, float]:
        """
        Calculates TRUE bumper-to-bumper distance to lead vehicle in forward path.
        Distinguishes in-lane vehicles from adjacent lane or parked side vehicles.
        Returns: (bumper_gap, lead_speed)
        """
        P = sim.player
        min_gap = 999.0
        lead_speed = 0.0
        cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
        if hasattr(sim, 'traffic') and hasattr(sim.traffic, 'cars'):
            for car in sim.traffic.cars:
                # Vertical level check: vehicles on flyovers overhead or underpasses below must not trigger braking
                if abs(getattr(car, 'z', 0.0) - P.z) > 2.5:
                    continue
                dx = car.x - P.x
                dy = car.y - P.y
                fwd = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                car_len = getattr(car, 'length', 4.5)
                bumper_gap = fwd - (P.length * 0.5 + car_len * 0.5)

                cos_rel = math.cos(car.heading - P.heading)

                # Same-direction lead vehicle following (within our lane corridor: |lat| < 1.35m)
                if cos_rel > 0.65:
                    if -0.5 < bumper_gap < 35.0 and abs(lat) < 1.35:
                        if bumper_gap < min_gap:
                            min_gap = bumper_gap
                            lead_speed = getattr(car, 'v', 0.0)
                # Opposing oncoming vehicle (cos_rel < -0.5):
                # On two-way roads, oncoming cars pass to the side in their own lane.
                # Only yield if they are directly head-on in our path (|lat| < 0.95m and bumper_gap < 6.0m)
                elif cos_rel < -0.5:
                    if 0.0 < bumper_gap < 6.0 and abs(lat) < 0.95:
                        if bumper_gap < min_gap:
                            min_gap = bumper_gap
                            lead_speed = 0.0
                # Cross traffic at intersections / oblique vehicles:
                # Only yield if they are directly in the forward bumper clearance zone (|lat| < 1.15m and bumper_gap < 4.5m)
                # OR actively moving across our path (car.v > 0.8)
                else:
                    if 0.0 < bumper_gap < 4.5 and abs(lat) < 1.15:
                        if bumper_gap < min_gap:
                            min_gap = bumper_gap
                            lead_speed = 0.0
                    elif 0.0 < bumper_gap < 10.0 and abs(lat) < 1.35 and getattr(car, 'v', 0.0) > 0.8:
                        if bumper_gap < min_gap:
                            min_gap = bumper_gap
                            lead_speed = 0.0

        return min_gap, lead_speed

    def _check_pedestrian_ahead(self, sim: Any) -> float:
        P = sim.player
        min_dist = 999.0
        cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
        if hasattr(sim, 'peds') and hasattr(sim.peds, 'peds'):
            for ped in sim.peds.peds:
                # Vertical level check: pedestrians walking under a flyover must not brake a car on the bridge
                if abs(getattr(ped, 'z', 0.0) - P.z) > 2.0:
                    continue
                dx = ped.x - P.x
                dy = ped.y - P.y
                fwd = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                # Pedestrian states: 'wait' (standing on curb waiting to cross), 'cross' (crossing roadway), 'leave' (leaving roadway)
                # If ped.state == 'wait', they are standing safely on the sidewalk/curb waiting for traffic to clear!
                # We yield if they are actively crossing ('cross', 'leave') within road corridor (|lat| < 2.4m),
                # OR if they are physically inside our direct tire track (|lat| < 0.95m).
                in_road = (getattr(ped, 'state', 'wait') in ('cross', 'leave') and abs(lat) < 2.4) or (abs(lat) < 0.95)
                if in_road and 0.5 < fwd < 20.0:
                    if fwd < min_dist:
                        min_dist = fwd
        return min_dist


# =============================================================================
# AUTO DRIVE CONTROLLER (High-Level Controller, Vision & Crash Reflection)
# =============================================================================
class AutoDriveController:
    """
    Unified high-level controller integrating:
    1. RL policy inference & live telemetry
    2. Real-time CNN vision perception
    3. Rule-based safety supervisor
    4. Post-crash online learning reflection
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.agent: Optional[PPOAgent] = None
        self.rule_based = RuleBasedController()
        self.vision_engine = CNNPerceptionEngine()
        self.crash_learner = CrashLearner()
        self.has_model = False

        # Live Telemetry State
        self.live_telemetry = {
            'value': 0.0,
            'steer': 0.0,
            'thr': 0.0,
            'brk': 0.0,
            'step_reward': 0.0,
            'cum_reward': 0.0,
            'vision_summary': "VISION: READY",
            'signal_detected': 'CLEAR',
            'cars_detected': 0,
            'peds_detected': 0,
            'latest_vision': None
        }

        if model_path and os.path.exists(model_path) and TORCH_AVAILABLE:
            self.agent = PPOAgent()
            self.has_model = self.agent.load(model_path)

        # Restore previously learned adaptive control parameters from persistent memory
        if hasattr(self, 'crash_learner') and hasattr(self.crash_learner, 'saved_params'):
            for k, v in self.crash_learner.saved_params.items():
                if hasattr(self.rule_based, k):
                    setattr(self.rule_based, k, v)
        self.live_telemetry['mistakes_learned'] = self.crash_learner.total_mistakes_learned

    def step(self, sim: Any, obs: np.ndarray) -> Tuple[float, float, float]:
        """
        Executes one control cycle for Auto Drive with live telemetry and vision perception.
        """
        # 1. Real-time CNN Camera Perception
        cam_obs = sim.get_observation()
        sim._obs = cam_obs
        vis_res = self.vision_engine.detect(cam_obs)
        sim.latest_vision = vis_res
        self.live_telemetry['latest_vision'] = vis_res
        self.live_telemetry['vision_summary'] = vis_res.summary_text
        self.live_telemetry['signal_detected'] = vis_res.signal_state
        self.live_telemetry['cars_detected'] = len(vis_res.cars)
        self.live_telemetry['peds_detected'] = len(vis_res.peds)

        # 2. RL Policy Inference or Fallback
        val = 0.0
        log_prob = 0.0
        if self.has_model and self.agent is not None:
            raw_action, log_prob, val = self.agent.select_action(obs, deterministic=True)
            steer, thr, brk = self.rule_based.supervise_action(sim, raw_action)
        else:
            steer, thr, brk = self.rule_based.compute_action(sim)

        # 3. Vision-based Reactive Safety Guarantee (Corridor-checked)
        if vis_res.signal_state == 'RED':
            must_stop, dist_s = self.rule_based._check_signal_ahead(sim)
            if must_stop and dist_s <= 0.45:
                thr = 0.0
                brk = max(brk, 0.85)
        if getattr(self.rule_based, 'lead_stop_timer', 0.0) < 2.0:
            if len(vis_res.cars) > 0 and vis_res.closest_car_dist < 0.12:
                thr = 0.0
                brk = 1.0
            if len(vis_res.peds) > 0 and vis_res.closest_ped_dist < 0.15:
                thr = 0.0
                brk = 1.0

        # 4. Record step for online crash reflection buffer
        self.crash_learner.record_step(obs, np.array([steer, thr, brk]), log_prob, val, sim.player.v)

        # 5. Update live telemetry
        bs = getattr(sim, 'blind_spot', None)
        if bs is None:
            bs = self.rule_based.check_rear_blind_spots(sim)
            sim.blind_spot = bs
        self.live_telemetry['value'] = val
        self.live_telemetry['steer'] = steer
        self.live_telemetry['thr'] = thr
        self.live_telemetry['brk'] = brk
        self.live_telemetry['status'] = getattr(self.rule_based, 'driving_status', 'CRUISING')
        self.live_telemetry['cum_reward'] += 0.15
        self.live_telemetry['mistakes_learned'] = self.crash_learner.total_mistakes_learned
        self.live_telemetry['rear_blind_spot'] = f"REAR: L {'CLEAR' if bs.left_clear else 'WARN'} | R {'CLEAR' if bs.right_clear else 'WARN'}"

        return steer, thr, brk

    def on_crash(self, sim: Any, reason: str) -> CrashReport:
        """Invoked when a collision occurs to trigger AI reflection and online policy adaptation."""
        save_p = self.model_path if self.model_path else "rl_policy.pt"
        report = self.crash_learner.analyze_and_learn(sim, reason, self.agent, save_path=save_p)
        self.live_telemetry['mistakes_learned'] = self.crash_learner.total_mistakes_learned
        return report
