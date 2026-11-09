"""
rl_env.py - Gym-Style Simulation Environment for Reinforcement Learning
========================================================================
Wraps the existing procedural simulator in a headless, fast-forward environment
adhering to standard Gym/Gymnasium conventions (reset, step, observation, reward).
Reuses the exact vehicle physics, traffic AI, pedestrian dynamics, and A* routing
from driving_sim.py without duplication.
"""

from typing import Tuple, Dict, Any, Optional, List
import math
import random
import numpy as np

import driving_sim
from driving_sim import Config, Simulator
from rl_agent import (
    R_PROGRESS, R_WAYPOINT, R_DESTINATION, R_LANE_CENTER, R_SPEED_MATCH,
    R_COLLISION, R_RED_LIGHT, R_SPEEDING, R_WRONG_WAY, R_MIDDLE_DIVIDER_CROSS, R_OFF_ROAD, R_ROAD_EDGE_CROSS,
    R_YIELD_FAIL, R_SMOOTHNESS, R_IDLE_PENALTY,
    R_LANE_STRADDLE, R_CURVE_SPEEDING, R_TAILGATE, R_FALSE_STOP
)


class CityDrivingRLEnv:
    """
    Continuous control environment for autonomous city driving.
    Observation dim: 28 floats (normalized)
    Action dim: 3 floats (steer [-1, 1], throttle [0, 1], brake [0, 1])
    """
    def __init__(self, seed: Optional[int] = None, traffic: int = 40,
                 frame_skip: int = 2, max_steps: int = 500,
                 vehicle: str = "sedan", curriculum_level: int = 1):
        self.base_seed = seed
        self.traffic_count = traffic
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.vehicle = vehicle
        self.curriculum_level = curriculum_level
        self.step_idx = 0
        self.episodes_run = 0

        # Construct underlying simulator in headless mode
        self.sim: Simulator = Simulator(
            model_path=None,
            seed=seed if seed is not None else 42,
            headless=True,
            target_fps=1000,
            spawn=0,
            traffic=traffic,
            vehicle=vehicle
        )

        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_nav_dist = 0.0
        self.prev_prog_idx = 0
        self.prev_stop_line_dist = 999.0
        self.passed_red_light = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.episodes_run += 1
        if seed is not None:
            self.base_seed = seed

        # Rebuild city periodically for diversity in higher curriculum levels
        if self.curriculum_level >= 2 and (self.episodes_run % 25 == 0):
            new_seed = random.randrange(1000, 999999) if self.base_seed is None else (self.base_seed + self.episodes_run)
            self.sim._build_world(new_seed)

        # Randomize spawn location
        self.sim.spawn_idx = random.randrange(0, len(self.sim.city.spawns))
        self.sim._respawn()

        # Randomize destination landmark
        dest_idx = random.randrange(0, len(self.sim.city.destinations))
        self.sim.select_destination(dest_idx)

        # Apply curriculum traffic settings
        if self.curriculum_level == 1:
            self.sim.traffic.cars.clear()
            self.sim.peds.peds.clear()
        elif self.curriculum_level == 2:
            self.sim.peds.peds.clear()

        # Randomize time of day occasionally
        if self.curriculum_level >= 3 and random.random() < 0.25:
            times = ["day", "sunset", "night"]
            self.sim.set_time_of_day(random.choice(times))

        self.step_idx = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_nav_dist = getattr(self.sim, 'nav_dist', 1000.0)
        self.prev_prog_idx = getattr(self.sim, 'nav_prog_idx', 0)
        self.prev_stop_line_dist = 999.0
        self.passed_red_light = False

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_idx += 1
        steer = float(np.clip(action[0], -1.0, 1.0))
        thr = float(np.clip(action[1], 0.0, 1.0))
        brk = float(np.clip(action[2], 0.0, 1.0))

        dt = Config.PHYS_DT
        for _ in range(self.frame_skip):
            self.sim._step(steer, thr, brk, dt)
            if self.sim.crashed or getattr(self.sim, 'mission_completed', False):
                break

        self.sim.update_navigation()

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward_and_status(steer, thr, brk, obs)
        truncated = self.step_idx >= self.max_steps

        self.prev_action = np.array([steer, thr, brk], dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        P = self.sim.player
        speed = abs(P.v)
        norm_speed = min(1.0, speed / 25.0)

        # 1. Road & Waypoint Geometry
        pts = getattr(self.sim, 'nav_route_pts', None)
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

        # Next turn curvature
        turn_angle = 0.0
        if len(pts) >= 3:
            wp2 = pts[min(2, len(pts) - 1)]
            angle2 = math.atan2(wp2[1] - wp[1], wp2[0] - wp[0])
            turn_angle = (angle2 - target_angle + math.pi) % (2.0 * math.pi) - math.pi
        norm_turn_angle = turn_angle / math.pi

        # Lateral lane offset & heading error
        res = self.sim.city.nearest_path(P.x, P.y, P.heading, P.z, maxd=10.0)
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

        # 2. Traffic Signal State
        sig_state_val = 0.0
        norm_stop_dist = 1.0
        if res is not None:
            lane, s_pos, _ = res
            if getattr(lane, 'signal', None) and getattr(lane, 'stop_s', None) is not None:
                d_stop = lane.stop_s - s_pos
                if 0.0 <= d_stop < 60.0:
                    norm_stop_dist = d_stop / 60.0
                    st = lane.signal.state(lane.axis, self.sim.sim_time)
                    if st == 'G':
                        sig_state_val = 1.0
                    elif st == 'Y':
                        sig_state_val = 0.2
                    elif st == 'R':
                        sig_state_val = -1.0

        # 3. Vehicle & Pedestrian Obstacles Ahead
        cos_h, sin_h = math.cos(P.heading), math.sin(P.heading)
        veh_ahead_dist = 60.0
        rel_speed = 0.0
        nearby_cars = [c for c in getattr(self.sim.traffic, 'cars', []) if abs(c.x - P.x) < 55.0 and abs(c.y - P.y) < 55.0]
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
        nearby_peds = [p for p in getattr(self.sim.peds, 'peds', []) if abs(p.x - P.x) < 40.0 and abs(p.y - P.y) < 40.0]
        for ped in nearby_peds:
            pdx, pdy = ped.x - P.x, ped.y - P.y
            fwd = pdx * cos_h + pdy * sin_h
            lat = -pdx * sin_h + pdy * cos_h
            if 0.5 < fwd < ped_ahead_dist and abs(lat) < 3.0:
                ped_ahead_dist = fwd
        norm_ped_dist = ped_ahead_dist / 40.0

        # 4. 16-Ray Lidar Array
        lidar = self._cast_lidar(nearby_cars, nearby_peds, num_rays=16, max_range=50.0)

        # Concatenate 28 features
        obs = np.array([
            norm_speed, norm_hd_err, norm_lat_off, norm_dist_wp, norm_bearing_wp,
            norm_turn_angle, norm_limit, sig_state_val, norm_stop_dist,
            norm_veh_dist, norm_rel_speed, norm_ped_dist, *lidar
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _cast_lidar(self, nearby_cars: List[Any], nearby_peds: List[Any],
                    num_rays: int = 16, max_range: float = 50.0) -> List[float]:
        P = self.sim.player
        rays = []
        angle_step = (2.0 * math.pi) / num_rays

        for i in range(num_rays):
            ray_angle = P.heading + i * angle_step
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            min_d = max_range
            for car in nearby_cars:
                dx = car.x - P.x
                dy = car.y - P.y
                proj = dx * ray_dx + dy * ray_dy
                if 0.5 < proj < min_d:
                    perp = abs(-dx * ray_dy + dy * ray_dx)
                    if perp < (car.width * 0.8):
                        min_d = proj

            for ped in nearby_peds:
                dx = ped.x - P.x
                dy = ped.y - P.y
                proj = dx * ray_dx + dy * ray_dy
                if 0.5 < proj < min_d:
                    perp = abs(-dx * ray_dy + dy * ray_dx)
                    if perp < 1.0:
                        min_d = proj

            rays.append(min_d / max_range)

        return rays

    def _compute_reward_and_status(self, steer: float, thr: float, brk: float,
                                   obs: np.ndarray) -> Tuple[float, bool, Dict[str, Any]]:
        P = self.sim.player
        info: Dict[str, Any] = {
            "progress": 0.0, "collision": False, "red_light": False,
            "speeding": False, "off_road": False, "destination": False
        }

        if self.sim.crashed:
            info["collision"] = True
            return R_COLLISION, True, info

        if getattr(self.sim, 'mission_completed', False):
            info["destination"] = True
            return R_DESTINATION, True, info

        total_reward = 0.0

        # Progress along A* route
        curr_nav_dist = getattr(self.sim, 'nav_dist', self.prev_nav_dist)
        dist_progress = max(0.0, self.prev_nav_dist - curr_nav_dist)
        total_reward += dist_progress * R_PROGRESS
        info["progress"] = dist_progress
        self.prev_nav_dist = curr_nav_dist

        # Waypoint clearance
        curr_prog_idx = getattr(self.sim, 'nav_prog_idx', self.prev_prog_idx)
        if curr_prog_idx > self.prev_prog_idx:
            total_reward += (curr_prog_idx - self.prev_prog_idx) * R_WAYPOINT
            self.prev_prog_idx = curr_prog_idx

        # Lane Centering & Mistake 1: Lane Straddling Penalty
        norm_lat_off = abs(obs[2])
        if norm_lat_off > 0.32:
            total_reward += R_LANE_STRADDLE * norm_lat_off
            info["lane_straddle"] = True
        else:
            total_reward += R_LANE_CENTER * max(0.0, 1.0 - norm_lat_off * 2.0)

        # Red Light Violation
        sig_state = obs[7]
        norm_stop_dist = obs[8]
        if sig_state < -0.5:
            if norm_stop_dist < 0.05 and self.prev_stop_line_dist >= 0.05:
                total_reward += R_RED_LIGHT
                info["red_light"] = True
        self.prev_stop_line_dist = norm_stop_dist

        # Speed limit compliance
        speed = abs(P.v)
        code, posted_lim = self.sim.city.surface_at(P.x, P.y, P.lvl)
        max_legal_speed = (posted_lim / 3.6) if (code in (1, 2) and posted_lim) else 11.1
        if speed > max_legal_speed + 1.5:
            over = speed - max_legal_speed
            total_reward += R_SPEEDING * over
            info["speeding"] = True
        elif speed > 2.0:
            total_reward += R_SPEED_MATCH * (speed / max_legal_speed)

        # Mistake 2: Curve Speeding Penalty (entering or navigating turns above 15 km/h)
        norm_turn_angle = abs(obs[5])
        if norm_turn_angle > 0.35 and speed > 4.2:
            total_reward += R_CURVE_SPEEDING * (speed - 4.0)
            info["curve_speeding"] = True

        # Mistake 3: Tailgating Penalty (too close to lead car while moving)
        norm_veh_dist = obs[9]
        if norm_veh_dist < 0.07 and speed > 2.0:
            total_reward += R_TAILGATE
            info["tailgate"] = True

        # Mistake: Wrong-Way & Middle Divider Crossing Penalties (Strict Incoming Line Avoidance)
        is_wrong_way = False
        is_middle_cross = False
        is_edge_cross = (code not in (1, 2, 3))
        res = self.sim.city.nearest_path(P.x, P.y, P.heading, P.z, maxd=6.0)
        if res is not None:
            lane = res[0]
            if getattr(lane, 'edge', None) is not None:
                e = lane.edge
                if not getattr(e.rt, 'oneway', False) and hasattr(e, 'cl') and hasattr(e, 'cum'):
                    best_d = 1e9
                    best_s = 0.0
                    for i in range(len(e.cl) - 1):
                        a, b = e.cl[i], e.cl[i + 1]
                        dx, dy = b[0] - a[0], b[1] - a[1]
                        l2 = dx * dx + dy * dy
                        t_val = 0.0 if l2 < 1e-9 else max(0.0, min(1.0, ((P.x - a[0]) * dx + (P.y - a[1]) * dy) / l2))
                        proj_x, proj_y = a[0] + t_val * dx, a[1] + t_val * dy
                        d = math.hypot(P.x - proj_x, P.y - proj_y)
                        if d < best_d:
                            best_d = d
                            best_s = e.cum[i] + (e.cum[i + 1] - e.cum[i]) * t_val
                    cx, cy, ch = e.pose(best_s)[:3]
                    lat_cl = -(P.x - cx) * math.sin(ch) + (P.y - cy) * math.cos(ch)
                    is_fwd = (lane.tdir == e.d)
                    travel_lat = lat_cl if is_fwd else -lat_cl
                    if travel_lat < -0.20:
                        is_middle_cross = True
                        if travel_lat < -0.90:
                            is_wrong_way = True
                    elif travel_lat > (e.rt.half + 0.15):
                        is_edge_cross = True

        if is_wrong_way:
            total_reward += R_WRONG_WAY
            info["wrong_way"] = True
        elif is_middle_cross:
            total_reward += R_MIDDLE_DIVIDER_CROSS
            info["middle_cross"] = True

        # Road edge / off-road check
        if is_edge_cross or abs(norm_lat_off) > 0.45:
            total_reward += R_ROAD_EDGE_CROSS
            info["off_road"] = True

        # Action smoothness
        action_diff = np.sum(np.abs(np.array([steer, thr, brk]) - self.prev_action))
        total_reward += R_SMOOTHNESS * action_diff

        # Mistake 4: False Stop Penalty (coming to standstill without signal or obstacle)
        norm_ped_dist = obs[11]
        if speed < 0.5 and sig_state >= 0.0 and norm_veh_dist > 0.20 and norm_ped_dist > 0.20:
            total_reward += R_FALSE_STOP
            info["false_stop"] = True
        elif speed < 0.5 and sig_state >= 0.0 and norm_veh_dist > 0.3:
            total_reward += R_IDLE_PENALTY

        return float(total_reward), False, info
