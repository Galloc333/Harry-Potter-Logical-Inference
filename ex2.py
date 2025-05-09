from collections import deque
import random
import math

ids = ['318931672','208302661']

class GringottsController:
    def __init__(self, world_dims, harry_loc, initial_clues):
        self.SYMBOLS = {
            'unexplored': -1,
            'corridor': 0,
            'dragon': 1,
            'vault': 2,
            'trap': 3,
            'vault_trap': 5,
            'dragon_trap': 6,
            'suspect_trap': 8,
            'suspect_trap_in_vault': 9,
            'retrieved_vault': 10,
            'suspect_trap_in_dragon': 11,
        }

        self.map_corners = {
            (0, 0),
            (0, world_dims[1] - 1),
            (world_dims[0] - 1, 0),
            (world_dims[0] - 1, world_dims[1] - 1),
        }

        self.next_to_corners = {
            (0, 1), (1, 0),
            (0, world_dims[1] - 2), (1, world_dims[1] - 1),
            (world_dims[0] - 2, 0), (world_dims[0] - 1, 1),
            (world_dims[0] - 2, world_dims[1] - 1), (world_dims[0] - 1, world_dims[1] - 2),
        }

        self.grid_rows = world_dims[0]
        self.grid_cols = world_dims[1]
        self.harry_current = harry_loc
        self.start_observations = initial_clues

        self.grid_map = [
            [self.SYMBOLS['unexplored'] for _ in range(self.grid_cols)]
            for _ in range(self.grid_rows)
        ]
        self.grid_map[harry_loc[0]][harry_loc[1]] = self.SYMBOLS['corridor']

        self.visited_spots = set()
        self.visited_spots.add(harry_loc)

        self.recent_moves = deque(maxlen=3)

        self.known_rows_set = set()
        self.known_cols_set = set()
        self.dragons_found = set()
        self.vaults_found = set()
        self.traps_possible = set()
        self.trap_destroy_attempts = {}
        self.vaults_collected = set()

        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = self.harry_current[0] + dx
            cc = self.harry_current[1] + dy
            if 0 <= rr < self.grid_rows and 0 <= cc < self.grid_cols:
                if (rr, cc) in self.map_corners:
                    self.map_corners.discard((rr, cc))
                    for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        r2 = rr + ddx
                        c2 = cc + ddy
                        if 0 <= r2 < self.grid_rows and 0 <= c2 < self.grid_cols:
                            if (r2, c2) in self.next_to_corners:
                                self.next_to_corners.discard((r2, c2))

        self.refresh_knowledge(initial_clues)

    def refresh_knowledge(self, clues):
        if not clues:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r1 = self.harry_current[0] + dx
                c1 = self.harry_current[1] + dy
                if 0 <= r1 < self.grid_rows and 0 <= c1 < self.grid_cols:
                    if self.grid_map[r1][c1] == self.SYMBOLS['unexplored']:
                        self.grid_map[r1][c1] = self.SYMBOLS['corridor']
        s_found = any(obs[0] == 'sulfur' for obs in clues)
        if not s_found:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = self.harry_current[0] + dx
                cc = self.harry_current[1] + dy
                if 0 <= rr < self.grid_rows and 0 <= cc < self.grid_cols:
                    self.traps_possible.discard((rr, cc))
                    val = self.grid_map[rr][cc]
                    if val == self.SYMBOLS['suspect_trap']:
                        self.grid_map[rr][cc] = self.SYMBOLS['corridor']
                    elif val == self.SYMBOLS['suspect_trap_in_vault']:
                        self.grid_map[rr][cc] = self.SYMBOLS['vault']
                    elif val == self.SYMBOLS['unexplored']:
                        self.grid_map[rr][cc] = self.SYMBOLS['corridor']
                    elif val == self.SYMBOLS['suspect_trap_in_dragon']:
                        self.grid_map[rr][cc] = self.SYMBOLS['dragon']
        susp_count = 0
        singled = None
        for obs in clues:
            if obs[0] == 'sulfur':
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = self.harry_current[0] + dx
                    cc = self.harry_current[1] + dy
                    if 0 <= rr < self.grid_rows and 0 <= cc < self.grid_cols:
                        if (rr, cc) not in self.trap_destroy_attempts:
                            self.traps_possible.add((rr, cc))
                            curv = self.grid_map[rr][cc]
                            if curv in [self.SYMBOLS['unexplored'], self.SYMBOLS['suspect_trap']]:
                                self.grid_map[rr][cc] = self.SYMBOLS['suspect_trap']
                                susp_count += 1
                                singled = (rr, cc)
                            elif curv == self.SYMBOLS['vault']:
                                self.grid_map[rr][cc] = self.SYMBOLS['suspect_trap_in_vault']
                                susp_count += 1
                                singled = (rr, cc)
                            elif curv == self.SYMBOLS['dragon']:
                                self.grid_map[rr][cc] = self.SYMBOLS['suspect_trap_in_dragon']
                                susp_count += 1
                                singled = (rr, cc)
                            elif curv in [
                                self.SYMBOLS['suspect_trap_in_dragon'],
                                self.SYMBOLS['dragon_trap'],
                                self.SYMBOLS['suspect_trap_in_vault'],
                                self.SYMBOLS['trap']
                            ]:
                                susp_count += 1
                                singled = (rr, cc)
                if susp_count == 1 and singled:
                    vv = self.grid_map[singled[0]][singled[1]]
                    if vv == self.SYMBOLS['suspect_trap']:
                        self.grid_map[singled[0]][singled[1]] = self.SYMBOLS['trap']
                    elif vv == self.SYMBOLS['suspect_trap_in_vault']:
                        self.grid_map[singled[0]][singled[1]] = self.SYMBOLS['vault_trap']
                    elif vv == self.SYMBOLS['suspect_trap_in_dragon']:
                        self.grid_map[singled[0]][singled[1]] = self.SYMBOLS['dragon_trap']
            elif obs[0] == 'vault':
                (rx, ry) = obs[1]
                if (rx, ry) not in self.vaults_found:
                    self.vaults_found.add((rx, ry))
                cv = self.grid_map[rx][ry]
                if cv in [self.SYMBOLS['unexplored'], self.SYMBOLS['corridor']]:
                    self.grid_map[rx][ry] = self.SYMBOLS['vault']
                elif cv == self.SYMBOLS['suspect_trap']:
                    self.grid_map[rx][ry] = self.SYMBOLS['suspect_trap_in_vault']
                elif cv == self.SYMBOLS['trap']:
                    self.grid_map[rx][ry] = self.SYMBOLS['vault_trap']
            elif obs[0] == 'dragon':
                (xx, yy) = obs[1]
                self.dragons_found.add((xx, yy))
                cr = self.grid_map[xx][yy]
                if cr in [self.SYMBOLS['unexplored'], self.SYMBOLS['corridor']]:
                    self.grid_map[xx][yy] = self.SYMBOLS['dragon']
                elif cr == self.SYMBOLS['suspect_trap']:
                    self.grid_map[xx][yy] = self.SYMBOLS['suspect_trap_in_dragon']
                elif cr == self.SYMBOLS['trap']:
                    self.grid_map[xx][yy] = self.SYMBOLS['dragon_trap']

    def neighbors_of(self, coord):
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = coord[0] + ddx
            nc = coord[1] + ddy
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                yield (nr, nc)

    def blocked_by_dragon(self, coord):
        val = self.grid_map[coord[0]][coord[1]]
        return val in [
            self.SYMBOLS['dragon'],
            self.SYMBOLS['dragon_trap'],
            self.SYMBOLS['suspect_trap_in_dragon']
        ]

    def bfs_explore(self, origin):
        scanned = set([origin])
        todo = deque([(origin, [origin])])
        distance_map = {origin: 0}
        routes = {origin: [origin]}
        while todo:
            spot, path = todo.popleft()
            for nxt in self.neighbors_of(spot):
                if nxt not in scanned and not self.blocked_by_dragon(nxt):
                    scanned.add(nxt)
                    cod = self.grid_map[nxt[0]][nxt[1]]
                    cost = 2 if cod in [
                        self.SYMBOLS['trap'],
                        self.SYMBOLS['vault_trap'],
                        self.SYMBOLS['suspect_trap'],
                        self.SYMBOLS['suspect_trap_in_vault']
                    ] else 1
                    distance_map[nxt] = distance_map[spot] + cost
                    r2 = path + [nxt]
                    routes[nxt] = r2
                    todo.append((nxt, r2))
        return distance_map, routes

    def search_path(self, origin, destinations):
        dmap, routes = self.bfs_explore(origin)
        best_dist = float('inf')
        top_coords = []
        for dpos in destinations:
            if dpos in dmap:
                dd = dmap[dpos]
                if dd < best_dist:
                    best_dist = dd
                    top_coords = [dpos]
                elif dd == best_dist:
                    top_coords.append(dpos)
        if not top_coords:
            return None
        cgoal = random.choice(top_coords)
        return routes[cgoal]

    def hunt_unexplored_zones(self):
        scanned = set()
        clusters = []
        for rr in range(self.grid_rows):
            for cc in range(self.grid_cols):
                if self.grid_map[rr][cc] == self.SYMBOLS['unexplored'] and (rr, cc) not in scanned:
                    check = deque([(rr, cc)])
                    scanned.add((rr, cc))
                    block = {(rr, cc)}
                    while check:
                        ap, bp = check.popleft()
                        for (vr, vc) in self.neighbors_of((ap, bp)):
                            if (vr, vc) not in scanned:
                                if 0 <= vr < self.grid_rows and 0 <= vc < self.grid_cols:
                                    if self.grid_map[vr][vc] == self.SYMBOLS['unexplored']:
                                        scanned.add((vr, vc))
                                        check.append((vr, vc))
                                        block.add((vr, vc))
                    clusters.append(block)
        return clusters

    def pick_unknown_spot(self, orig):
        dmap, routes = self.bfs_explore(orig)
        unknowns = self.hunt_unexplored_zones()
        bag = []
        for clt in unknowns:
            size_ = len(clt)
            chosen = None
            bestd = float('inf')
            for coords in clt:
                if coords in dmap:
                    dist_ = dmap[coords]
                    if dist_ < bestd:
                        bestd = dist_
                        chosen = coords
            if chosen:
                score = size_ / (1 + bestd)
                bag.append((score, size_, bestd, chosen))
        if not bag:
            return None, None
        bag.sort(key=lambda x: x[0], reverse=True)
        top_s = bag[0][0]
        ties = [x for x in bag if x[0] == top_s]
        _, _, _, picked = random.choice(ties)
        _, routes = self.bfs_explore(orig)
        if picked in routes:
            return picked, routes[picked]
        return None, None

    def corner_steps(self):
        adj_cells = list(self.neighbors_of(self.harry_current))
        pot = [p for p in adj_cells if p in self.next_to_corners]
        if len(pot) == 1:
            nxt = pot[0]
            if self.blocked_by_dragon(nxt):
                return None
            if self.handle_trap(nxt):
                return ("destroy", nxt)
            self.do_action("move", nxt)
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                rx = nxt[0] + dx
                ry = nxt[1] + dy
                if (rx, ry) in self.map_corners:
                    self.map_corners.discard((rx, ry))
                    for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rx2 = rx + ddx
                        ry2 = ry + ddy
                        if (rx2, ry2) in self.next_to_corners:
                            self.next_to_corners.discard((rx2, ry2))
            return ("move", nxt)
        elif len(pot) == 2:
            route = self.search_path(self.harry_current, self.map_corners)
            if route and len(route) > 1:
                ncell = route[1]
                if self.handle_trap(ncell):
                    return ("destroy", ncell)
                self.do_action("move", ncell)
                last_ = route[-1]
                self.map_corners.discard(last_)
                lx, ly = last_
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rx2 = lx + ddx
                    ry2 = ly + ddy
                    if (rx2, ry2) in self.next_to_corners:
                        self.next_to_corners.discard((rx2, ry2))
                return ("move", ncell)

    def handle_trap(self, place):
        val = self.grid_map[place[0]][place[1]]
        if val in [
            self.SYMBOLS['trap'],
            self.SYMBOLS['vault_trap'],
            self.SYMBOLS['suspect_trap'],
            self.SYMBOLS['suspect_trap_in_vault']
        ]:
            if place not in self.trap_destroy_attempts:
                self.trap_destroy_attempts[place] = 0
            if self.trap_destroy_attempts[place] < 1:
                self.trap_destroy_attempts[place] += 1
                self.do_action("destroy", place)
                return True
        return False

    def do_action(self, act, target=None):
        if act == "move":
            self.harry_current = target
            self.visited_spots.add(target)
            self.recent_moves.append(target)
        elif act == "destroy":
            self.traps_possible.discard(target)
            code_ = self.grid_map[target[0]][target[1]]
            if code_ == self.SYMBOLS['suspect_trap']:
                self.grid_map[target[0]][target[1]] = self.SYMBOLS['corridor']
            elif code_ == self.SYMBOLS['suspect_trap_in_vault']:
                self.grid_map[target[0]][target[1]] = self.SYMBOLS['vault']
            elif code_ == self.SYMBOLS['trap']:
                self.grid_map[target[0]][target[1]] = self.SYMBOLS['corridor']
            elif code_ == self.SYMBOLS['vault_trap']:
                self.grid_map[target[0]][target[1]] = self.SYMBOLS['vault']
            elif code_ == self.SYMBOLS['dragon_trap']:
                self.grid_map[target[0]][target[1]] = self.SYMBOLS['dragon']
        elif act == "collect":
            pass
        elif act == "wait":
            pass

    def get_next_action(self, sightings):
        self.refresh_knowledge(sightings)
        if (self.harry_current in self.vaults_found) and (self.harry_current not in self.vaults_collected):
            self.vaults_collected.add(self.harry_current)
            self.grid_map[self.harry_current[0]][self.harry_current[1]] = self.SYMBOLS['retrieved_vault']
            self.do_action("collect", self.harry_current)
            return ("collect",)
        vaults_left = [v for v in self.vaults_found if v not in self.vaults_collected]
        if vaults_left:
            maybe_susp = False
            for nb in self.neighbors_of(self.harry_current):
                xx, yy = nb
                if self.grid_map[xx][yy] == self.SYMBOLS['suspect_trap_in_vault']:
                    maybe_susp = True
            if maybe_susp:
                for nb in self.neighbors_of(self.harry_current):
                    xx, yy = nb
                    if self.grid_map[xx][yy] == self.SYMBOLS['vault_trap']:
                        self.handle_trap(nb)
                        return ("destroy", nb)
        if vaults_left:
            dests = [
                v for v in vaults_left
                if v in self.trap_destroy_attempts and self.trap_destroy_attempts[v] >= 1
            ]
            if dests:
                route = self.search_path(self.harry_current, set(dests))
            else:
                route = self.search_path(self.harry_current, set(vaults_left))
            if route and len(route) > 1:
                nx = route[1]
                if self.handle_trap(nx):
                    return ("destroy", nx)
                self.do_action("move", nx)
                return ("move", nx)
        corner_opt = self.corner_steps()
        if corner_opt:
            return corner_opt
        pick, route_ = self.pick_unknown_spot(self.harry_current)
        if pick and route_ and len(route_) > 1:
            nx = route_[1]
            if self.handle_trap(nx):
                return ("destroy", nx)
            self.do_action("move", nx)
            return ("move", nx)
        self.do_action("wait")
        return ("wait",)
