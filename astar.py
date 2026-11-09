"""
astar.py - Standalone A* Pathfinding Algorithm for City Driving Simulation
Finds the optimal shortest-distance road network path connecting vehicles to destinations.
"""

import math
import heapq


class AStarRouter:
    """
    Dedicated A* road network planner.
    Operates on directed lane and intersection connector graph to guarantee
    the global shortest-distance route to destination landmarks.
    """

    def __init__(self, city_map):
        self.city = city_map

    def find_approach_lanes(self, dest):
        """
        Finds all candidate street lanes bordering the destination landmark.
        Includes all lanes on dest['edge'] (both driving directions) and all perimeter lanes within 45m.
        Returns a dict: {lane.pid: (lane, best_s, curb_x, curb_y, dist_to_curb)}
        """
        cands = {}
        dest_pos = (dest['x'], dest['y'])
        curb_x, curb_y = dest.get('curb_pt', dest_pos)

        # 1. Primary road edge bordering the landmark (strictly legal approach direction)
        e = dest.get('edge')
        app_lane = dest.get('lane')
        if e is not None and hasattr(e, 'lanes'):
            allowed_dirs = [app_lane.tdir] if app_lane and hasattr(app_lane, 'tdir') else list(e.lanes.keys())
            for d_key in allowed_dirs:
                lane_list = e.lanes.get(d_key, [])
                for l in lane_list:
                    if getattr(l.edge, 'elev', None) is not None:
                        continue
                    best_d = 1e9
                    best_s = 0.0
                    for k in range(0, int(l.length) + 1):
                        px, py, _ = l.pose(float(k))
                        d = math.hypot(px - curb_x, py - curb_y)
                        if d < best_d:
                            best_d, best_s = d, float(k)
                    cx, cy, _ = l.pose(best_s)
                    cands[l.pid] = (l, best_s, cx, cy, best_d)

        # 2. If no direct edge candidate found, check perimeter lanes within 45m
        if not cands:
            for l in getattr(self.city, 'main_lanes', self.city.lanes):
                if getattr(l.edge, 'elev', None) is not None or l.pid in cands:
                    continue
                best_d = 1e9
                best_s = 0.0
                for k in range(0, int(l.length) + 1, 2):
                    px, py, _ = l.pose(float(k))
                    d = math.hypot(px - curb_x, py - curb_y)
                    if d < best_d:
                        best_d, best_s = d, float(k)
                if best_d <= 45.0:
                    cx, cy, _ = l.pose(best_s)
                    cands[l.pid] = (l, best_s, cx, cy, best_d)

        return cands

    def plan_optimal_route(self, car_x, car_y, car_heading, car_z, dest):
        """
        A* search over the directed road network to find the absolute shortest distance route.
        Returns: (route_segments, target_s, curb_pt) or (None, None, None)
        """
        dest_pos = (dest['x'], dest['y'])
        curb_pt = dest.get('curb_pt', dest_pos)

        # Direct arrival check: if already very close (< 25m) to destination
        if math.hypot(dest['x'] - car_x, dest['y'] - car_y) < 25.0:
            return [], None, curb_pt

        goal_cands = self.find_approach_lanes(dest)
        if not goal_cands:
            return None, None, None

        # Determine vehicle's current road lane and position
        res = self.city.nearest_path(car_x, car_y, car_heading, car_z, maxd=25.0)
        if res is None:
            res = self.city.nearest_path(car_x, car_y, car_heading, car_z, maxd=60.0)
        if res is None:
            # Fallback: find any nearest drivable lane regardless of heading
            bd = 1e9
            best_l = None
            best_s = 0.0
            for l in getattr(self.city, 'main_lanes', self.city.lanes):
                if getattr(l.edge, 'elev', None) is not None:
                    continue
                for k in range(0, int(l.length) + 1, 3):
                    px, py, _ = l.pose(float(k))
                    d = math.hypot(px - car_x, py - car_y)
                    if d < bd:
                        bd, best_l, best_s = d, l, float(k)
            if best_l is not None:
                res = (best_l, best_s, bd)

        if res is None:
            return None, None, None

        start_lane, start_s, _ = res

        # 1. Direct path check: Is the car currently on an approach lane ahead of destination?
        if start_lane.pid in goal_cands:
            l_info = goal_cands[start_lane.pid]
            goal_s = l_info[1]
            if start_s <= goal_s + 4.0:
                # Direct route straight ahead on this lane!
                return [start_lane], goal_s, (l_info[2], l_info[3])

        # 2. A* Priority Queue Search
        # g_dist[pid]: shortest known road distance from vehicle to the START of segment pid
        g_dist = {}
        came_from = {}
        pq = []
        count = 0

        best_goal = None
        best_goal_cost = float('inf')

        def h_val(p):
            # Admissible Euclidean heuristic from start of p to dest_pos
            sx, sy = p.pts[0][0], p.pts[0][1]
            return math.hypot(sx - dest_pos[0], sy - dest_pos[1])

        # Distance from vehicle's current longitudinal position to end of start_lane
        rem_start = max(0.0, start_lane.length - start_s)
        g_dist[start_lane.pid] = 0.0
        seen = {start_lane.pid}
        for conn in start_lane.out:
            if conn is None:
                continue
            g_dist[conn.pid] = rem_start
            came_from[conn.pid] = start_lane
            count += 1
            heapq.heappush(pq, (rem_start + h_val(conn), count, conn.pid, conn))

        while pq:
            f, _, pid, cur = heapq.heappop(pq)
            if pid in seen:
                continue
            seen.add(pid)

            cur_g = g_dist[pid]
            # Pruning: if current path cost exceeds best known complete route, terminate search
            if cur_g >= best_goal_cost:
                break

            # If cur is an approach lane, evaluate complete path cost to destination
            if cur.kind == 'lane' and cur.pid in goal_cands:
                l_info = goal_cands[cur.pid]
                goal_s = l_info[1]
                d_curb = l_info[4]
                # Total driving distance = (distance to start of cur) + (distance along cur to goal_s) + (distance to curb)
                total_cost = cur_g + goal_s + d_curb
                if total_cost < best_goal_cost:
                    best_goal_cost = total_cost
                    # Trace back route segments
                    r = [cur]
                    curr_b = cur.pid
                    seen_back = {curr_b}
                    while curr_b in came_from:
                        p_prev = came_from[curr_b]
                        if p_prev.pid in seen_back:
                            break
                        seen_back.add(p_prev.pid)
                        r.insert(0, p_prev)
                        curr_b = p_prev.pid
                        if curr_b == start_lane.pid:
                            break
                    best_goal = (r, goal_s, (l_info[2], l_info[3]))

            # Successors
            nxts = cur.out if cur.kind == 'lane' else ([cur.out_lane] if cur.out_lane else [])
            cost_to_end = cur_g + cur.length
            for q in nxts:
                if q is None or q.pid in seen:
                    continue
                step_cost = cost_to_end
                if step_cost < g_dist.get(q.pid, float('inf')):
                    g_dist[q.pid] = step_cost
                    came_from[q.pid] = cur
                    count += 1
                    heapq.heappush(pq, (step_cost + h_val(q), count, q.pid, q))

        if best_goal is not None:
            return best_goal

        return None, None, None

    def plan_route(self, start_paths, goal_candidates, dest_pos):
        """Backwards compatibility wrapper."""
        if not start_paths or not goal_candidates:
            return None, None, None
        sp = start_paths[0]
        pts = getattr(sp, 'pts', None)
        cx = pts[0][0] if pts else 0.0
        cy = pts[0][1] if pts else 0.0
        hd = getattr(sp, 'hd', None)
        heading = hd[0] if hd else 0.0
        return self.plan_optimal_route(cx, cy, heading, 0.0, {'x': dest_pos[0], 'y': dest_pos[1]})

    def build_polyline(self, car_x, car_y, route, end_s, curb_pt, dest_pt=None, approach_pts=None):
        """
        Assembles a clean, contiguous polyline that stays strictly on the road asphalt
        and terminates completely at the destination landmark pin / mission finish bay.
        """
        if not route:
            pts = [(car_x, car_y)]
            if curb_pt:
                pts.append(curb_pt)
            if approach_pts:
                for apt in approach_pts:
                    if apt and apt != curb_pt:
                        pts.append(apt)
            elif dest_pt and dest_pt != curb_pt:
                pts.append(dest_pt)
            return pts

        raw_pts = [(car_x, car_y)]

        # 1. Starting segment: include points ahead of the vehicle
        first_p = route[0]
        if len(route) == 1:
            if len(first_p.pts) > 1:
                best_k = 0
                best_kd = 1e9
                for k, (kx, ky) in enumerate(first_p.pts):
                    kd = (kx - car_x) ** 2 + (ky - car_y) ** 2
                    if kd < best_kd:
                        best_kd, best_k = kd, k
                for k in range(best_k + 1, len(first_p.pts)):
                    if end_s is not None and first_p.cum is not None and k < len(first_p.cum):
                        if first_p.cum[k] > end_s:
                            break
                    raw_pts.append(first_p.pts[k])
            else:
                raw_pts.extend(first_p.pts)
        else:
            if len(first_p.pts) > 1:
                best_k = 0
                best_kd = 1e9
                for k, (kx, ky) in enumerate(first_p.pts):
                    kd = (kx - car_x) ** 2 + (ky - car_y) ** 2
                    if kd < best_kd:
                        best_kd, best_k = kd, k
                raw_pts.extend(first_p.pts[best_k + 1:])
            else:
                raw_pts.extend(first_p.pts)

            # 2. Intermediate segments
            for p in route[1:-1]:
                raw_pts.extend(p.pts)

            # 3. Last segment: only include points up to the arrival curb
            last_p = route[-1]
            if end_s is not None and last_p.cum is not None:
                for k, pt in enumerate(last_p.pts):
                    if k < len(last_p.cum) and last_p.cum[k] <= end_s:
                        raw_pts.append(pt)
            else:
                raw_pts.extend(last_p.pts)

        # Road curb outside landmark
        if curb_pt:
            raw_pts.append(curb_pt)

        # Connect through entrance driveway into the mission finish bay
        if approach_pts:
            for apt in approach_pts:
                if apt and apt != curb_pt:
                    raw_pts.append(apt)
        elif dest_pt and dest_pt != curb_pt:
            raw_pts.append(dest_pt)

        # Remove duplicate / near-identical adjacent points (< 0.6m) and subdivide long segments (< 6m)
        clean_pts = [raw_pts[0]]
        for pt in raw_pts[1:]:
            last = clean_pts[-1]
            dist_sq = (pt[0] - last[0]) ** 2 + (pt[1] - last[1]) ** 2
            if dist_sq < 0.36:
                continue
            dist = math.sqrt(dist_sq)
            if dist > 6.0:
                num_sub = int(math.ceil(dist / 5.0))
                for step_i in range(1, num_sub):
                    frac = step_i / float(num_sub)
                    clean_pts.append((last[0] + (pt[0] - last[0]) * frac,
                                      last[1] + (pt[1] - last[1]) * frac))
            clean_pts.append(pt)

        return clean_pts
