import salabim as sim
import math
import random
import heapq
import pandas as pd

env = sim.Environment(trace=False, width=1300, height=950)
env.animate(True)
#sim.random_seed(42)

# =========================================================================
# 1. Setup & Layout Configuration
# =========================================================================

def run_simulation():
    global machines, mean_util_monitor, amr_fleets, all_amrs, machine_positions
    
    # --- CONFIGURATION (CHANGE THIS VALUE) ---
    NUM_AREAS = 10
    # -----------------------------------------

    TOTAL_MACHINES = 40
    TOTAL_AMRS = 10
    
    # Helper to distribute counts evenly, spreading the remainder RANDOMLY
    def get_distribution(total, num_areas):
        base = total // num_areas
        rem = total % num_areas
        dist = [base] * num_areas
        
        # Randomly select 'rem' unique areas to receive the extra unit
        extra_indices = random.sample(range(num_areas), rem)
        for idx in extra_indices:
            dist[idx] += 1
            
        return dist

    machine_dist = get_distribution(TOTAL_MACHINES, NUM_AREAS)
    amr_dist = get_distribution(TOTAL_AMRS, NUM_AREAS)

    machine_positions = []
    # Column 1
    y_starts_col1 = [75, 125, 175, 225, 325, 375, 425, 475, 525, 575, 625, 675, 725, 775, 825]
    for y in y_starts_col1: machine_positions.append((130, y))
    # Column 2
    y_starts_col2 = [825, 725, 625, 525, 425, 325, 225, 125, 75]
    for y in y_starts_col2: machine_positions.append((280, y))
    # Column 3
    y_starts_col3 = [75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 625, 675, 725, 775, 825]
    for y in y_starts_col3: machine_positions.append((430, y))
    
    # Sort positions by Y coordinate so areas are naturally grouped geographically (top to bottom)
    machine_positions.sort(key=lambda pos: pos[1])

    # =========================================================================
    # 2. Graph Architecture
    # =========================================================================
    global graph
    graph = {}
    def add_edge(n1, n2):
        if n1 not in graph: graph[n1] = []
        if n2 not in graph: graph[n2] = []
        if n2 not in graph[n1]: graph[n1].append(n2)
        if n1 not in graph[n2]: graph[n2].append(n1)

    x_lanes = [180, 230, 330, 380, 480]
    y_levels = [50, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 625, 675, 725, 775, 825, 875]

    for x in x_lanes:
        for i in range(len(y_levels) - 1): add_edge((x, y_levels[i]), (x, y_levels[i+1]))
    for y in [175, 275, 375, 475, 575, 675, 775]:
        add_edge((180, y), (230, y))
        add_edge((230, y), (330, y))
        add_edge((330, y), (380, y))
    for y in [75, 125, 225, 325, 425, 525, 625, 725, 825]:
        add_edge((180, y), (230, y))
        add_edge((330, y), (380, y))
    
    add_edge((180, 875), (230, 875))
    add_edge((230, 875), (330, 875))
    add_edge((330, 875), (380, 875))
    add_edge((380, 875), (480, 875))

    # Return lanes
    add_edge((480, 50), (480, 40))
    add_edge((480, 40), (450, 40))
    add_edge((450, 910), (480, 910))
    add_edge((480, 910), (480, 875))

    # AMR Parking Slots Configuration
    park_x_positions_top = [200, 250, 300, 350, 400]
    park_x_positions_bottom = [200, 250, 300, 350, 400]
    
    for px in park_x_positions_top:
        add_edge((px, 12), (px, 40))
        add_edge((px, 40), (450, 40))
        add_edge((px, 40), (px, 50))
        closest_x = min([180, 230, 330, 380], key=lambda x: abs(x - px))
        add_edge((px, 50), (closest_x, 50))

    for px in park_x_positions_bottom:
        add_edge((px, 938), (px, 910))
        add_edge((px, 910), (450, 910))
        add_edge((px, 910), (px, 900))
        closest_x = min([180, 230, 330, 380], key=lambda x: abs(x - px))
        add_edge((px, 900), (closest_x, 900))
        add_edge((closest_x, 900), (closest_x, 875))

    # =========================================================================
    # GLOBAL OCCUPANCY GRID
    # =========================================================================
    global global_occupied
    global_occupied = {}
    
    # Initialize AMR Fleets dynamically based on NUM_AREAS
    amr_fleets = {i: [] for i in range(1, NUM_AREAS + 1)}
    all_amrs = []

    def find_path(start, end, exclude_return_lane, self_amr):
        queue = [(0, start, [start])]
        seen = set()
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in seen: continue
            seen.add(node)
            if node == end: return path
            for nxt in graph.get(node, []):
                if nxt in seen: continue
                if exclude_return_lane:
                    if nxt == (450, 910) and node != (450, 910): continue
                    if nxt[1] >= 910 and node[1] < 910: continue
                    if nxt[0] == 480 and node[0] != 480: continue
                    if nxt == (450, 40) and node != (450, 40): continue
                    if nxt[1] <= 40 and node[1] > 40: continue
                else:
                    if node[0] == 480 and nxt[0] < 480 and 50 <= nxt[1] <= 875: continue
                    if end[1] <= 40:
                        if node[0] == 480 and nxt[0] == 480 and nxt[1] > node[1]: continue
                        if node[0] == 480 and nxt[0] < 480 and nxt[1] >= 910: continue
                    elif end[1] >= 910:
                        if node[0] == 480 and nxt[0] == 480 and nxt[1] < node[1]: continue
                        if node[0] == 480 and nxt[0] < 480 and nxt[1] <= 40: continue
                
                weight = 1
                if nxt in global_occupied and global_occupied[nxt] != self_amr:
                    weight += 10000
                heapq.heappush(queue, (cost + weight, nxt, path + [nxt]))
        return [start]

    def get_machine_stop_pos(idx):
        mx, my = machine_positions[idx]
        if mx == 130: return (180, my)
        elif mx == 280: return (230, my)
        else: return (380, my)

    # =========================================================================
    # 3. Machine Logic 
    # =========================================================================
    class Machine(sim.Component):
        def __init__(self, idx, assigned_area, *args, **kwargs):
            self.idx = idx
            self.area = assigned_area
            self.machine_state = "IDLE"
            self.raw_qty = 0
            self.processed_qty = 0
            self.process_time = 0.0
            self.total_idle = 0.0
            self.total_run = 0.0
            self.time_options = [9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
            sim.Component.__init__(self, *args, **kwargs)

        def check_and_send_request(self):
            assigned_amr_fleet = amr_fleets[self.area]
            is_queued = False
            for a in assigned_amr_fleet:
                if a.active:
                    if self.idx in a.request_queue or a.current_target == self.idx:
                        is_queued = True
                        break

            if not is_queued:
                available_amrs = [a for a in assigned_amr_fleet if a.active and not a.needs_charge]
                if available_amrs:
                    best_amr = min(available_amrs, key=lambda a: len(a.request_queue))
                    if self.idx not in best_amr.request_queue:
                        best_amr.request_queue.append(self.idx)

        def process(self):
            while True:
                self.machine_state = "IDLE"
                while self.raw_qty == 0:
                    self.check_and_send_request()
                    self.hold(0.1)
                    self.total_idle += 0.1

                self.machine_state = "BUSY"
                self.raw_qty -= 1
                self.process_time = random.choice(self.time_options)
                self.t_start_run = env.now()
                self.hold(self.process_time)
                self.total_run += (env.now() - self.t_start_run)
                
                self.processed_qty += 1
                self.machine_state = "COMPLETED"
                self.process_time = 0.0

                while self.processed_qty > 0:
                    self.check_and_send_request()
                    self.hold(0.1)
                    self.total_idle += 0.1

    # Instantiate Machines dynamically divided by area
    global machines
    machines = []
    current_m_idx = 0
    for area_idx, count in enumerate(machine_dist):
        area_id = area_idx + 1
        for _ in range(count):
            machines.append(Machine(idx=current_m_idx, assigned_area=area_id))
            current_m_idx += 1

    # =========================================================================
    # 4. AMR Logic (Upgraded Movement Engine)
    # =========================================================================
    class AMR(sim.Component):
        def __init__(self, idx, px, py, is_top_parking, assigned_area, *args, **kwargs):
            self.idx = idx
            self.x, self.y = px, py
            self.parking_x, self.parking_y = px, py
            self.is_top_parking = is_top_parking
            self.area = assigned_area
            self.raw_held = 0
            self.processed_held = 0
            self.items_handled = 0
            
            # Dynamic Colors for Areas
            amr_colors = ["royalblue", "firebrick", "darkorange", "purple", "teal", "magenta", "olive","tan","80%gray","darkturquoise"]
            self.color = amr_colors[(self.area - 1) % len(amr_colors)]
            
            self.active = True
            self.amr_state = "CHARGING"
            self.needs_charge = False
            self.request_queue = []
            self.current_target = None
            self.is_moving = False
            self.move_start_time = 0
            self.move_duration = 0
            self.x_from, self.y_from = px, py
            self.x_to, self.y_to = px, py
            self.transfer_type = None
            self.transfer_start = 0
            self.tx_target, self.ty_target = px, py
            sim.Component.__init__(self, *args, **kwargs)

        def move_to(self, target_node):
            is_return = (target_node[0] == 480 or target_node == (450, 40) or target_node[1] == 12 or target_node == (450, 910) or target_node[1] == 938)
            stuck_attempts = 0
            while (self.x, self.y) != target_node:
                path = find_path((self.x, self.y), target_node, exclude_return_lane=not is_return, self_amr=self)
                if len(path) <= 1:
                    self.amr_state = "WAITING"
                    self.hold(random.uniform(0.3, 0.7))
                    stuck_attempts += 1
                    continue
                next_node = path[1]
                if next_node in global_occupied and global_occupied[next_node] != self:
                    stuck_attempts += 1
                    if stuck_attempts < 4:
                        self.amr_state = "WAITING"
                        self.hold(random.uniform(0.1, 0.4))
                        continue
                    else:
                        yield_node = None
                        neighbors = list(graph.get((self.x, self.y), []))
                        random.shuffle(neighbors)
                        for neighbor in neighbors:
                            if neighbor not in global_occupied and neighbor != next_node:
                                if (not is_return) and (neighbor[0] == 480 or neighbor == (450, 40) or neighbor == (450, 910)):
                                    continue
                                yield_node = neighbor
                                break
                        if yield_node:
                            if (self.x, self.y) in global_occupied and global_occupied[(self.x, self.y)] == self:
                                del global_occupied[(self.x, self.y)]
                            global_occupied[yield_node] = self
                            self.x_from, self.y_from = self.x, self.y
                            self.x_to, self.y_to = yield_node
                            dist = math.hypot(self.x_to - self.x_from, self.y_to - self.y_from)
                            self.move_duration = dist / 180.0
                            self.move_start_time = env.now()
                            self.is_moving = True
                            self.amr_state = "YIELDING"
                            self.hold(self.move_duration)
                            self.x, self.y = self.x_to, self.y_to
                            self.is_moving = False
                            stuck_attempts = 0
                        else:
                            self.hold(0.5)
                        continue

                if (self.x, self.y) in global_occupied and global_occupied[(self.x, self.y)] == self:
                    del global_occupied[(self.x, self.y)]
                global_occupied[next_node] = self
                stuck_attempts = 0  
                self.x_from, self.y_from = self.x, self.y
                self.x_to, self.y_to = next_node
                dist = math.hypot(self.x_to - self.x_from, self.y_to - self.y_from)
                self.move_duration = dist / 180.0
                self.move_start_time = env.now()
                self.is_moving = True
                self.amr_state = "MOVING"
                self.hold(self.move_duration)
                self.x, self.y = self.x_to, self.y_to
                self.is_moving = False

        def process(self):
            global_occupied[(self.x, self.y)] = self
            if not self.active:
                while True: self.hold(1.0)
            
            self.hold(self.idx * 3.0)

            while True:
                if self.items_handled >= 20:
                    self.needs_charge = True
                    self.amr_state = "MOVING"
                    self.request_queue.clear()
                    
                    if self.processed_held > 0:
                        self.move_to((180, 875))
                        self.amr_state = "TRANSFERRING"
                        self.tx_target, self.ty_target = 130, 875
                        self.transfer_start = env.now()
                        self.transfer_type = "DELIVER"
                        self.hold(0.6)
                        self.transfer_type = None
                        self.processed_held = 0
                        
                    self.move_to((380, 875))
                    self.move_to((480, 875))
                    
                    if self.is_top_parking:
                        self.move_to((480, 40))
                        self.move_to((450, 40))
                        self.move_to((self.parking_x, 40))
                    else:
                        self.move_to((480, 910))
                        self.move_to((450, 910))
                        self.move_to((self.parking_x, 910))
                        
                    self.move_to((self.parking_x, self.parking_y))
                    self.amr_state = "CHARGING"
                    self.hold(15.0)
                    self.items_handled = 0
                    self.needs_charge = False
                    continue

                if self.processed_held >= 10:
                    self.amr_state = "MOVING"
                    self.move_to((180, 875))
                    self.amr_state = "TRANSFERRING"
                    self.tx_target, self.ty_target = 130, 875
                    self.transfer_start = env.now()
                    self.transfer_type = "DELIVER"
                    self.hold(0.6)
                    self.transfer_type = None
                    self.items_handled += self.processed_held
                    self.processed_held = 0
                    continue

                if len(self.request_queue) > 0:
                    target_idx = None
                    task_type = None
                    for m_idx in self.request_queue:
                        m = machines[m_idx]
                        if m.processed_qty > 0 and (self.raw_held + self.processed_held < 10):
                            target_idx = m_idx
                            task_type = "PICKUP"
                            break
                        elif m.machine_state == "IDLE" and m.raw_qty == 0 and self.raw_held > 0:
                            target_idx = m_idx
                            task_type = "DELIVER"
                            break
                        elif m.machine_state == "IDLE" and m.raw_qty == 0 and self.raw_held == 0 and (self.raw_held + self.processed_held < 10):
                            target_idx = m_idx
                            task_type = "FETCH_AND_DELIVER"
                            break

                    if target_idx is not None:
                        self.current_target = target_idx
                        if target_idx in self.request_queue:
                            self.request_queue.remove(target_idx)
                        mach = machines[target_idx]
                        stop_pt = get_machine_stop_pos(target_idx)
                        mx, my = machine_positions[target_idx]
                        
                        if task_type == "FETCH_AND_DELIVER":
                            self.move_to((180, 275))
                            self.amr_state = "TRANSFERRING"
                            self.tx_target, self.ty_target = 130, 275
                            self.transfer_start = env.now()
                            self.transfer_type = "PICKUP"
                            self.hold(0.6)
                            self.transfer_type = None
                            self.raw_held = 10 - self.processed_held
                            self.move_to(stop_pt)
                            self.amr_state = "TRANSFERRING"
                            self.tx_target, self.ty_target = mx, my
                            self.transfer_start = env.now()
                            self.transfer_type = "DELIVER"
                            self.hold(0.6)
                            self.transfer_type = None
                            mach.raw_qty += 1
                            self.raw_held -= 1
                            self.items_handled += 1
                        elif task_type == "DELIVER":
                            self.move_to(stop_pt)
                            self.amr_state = "TRANSFERRING"
                            self.tx_target, self.ty_target = mx, my
                            self.transfer_start = env.now()
                            self.transfer_type = "DELIVER"
                            self.hold(0.6)
                            self.transfer_type = None
                            mach.raw_qty += 1
                            self.raw_held -= 1
                            self.items_handled += 1
                        elif task_type == "PICKUP":
                            self.move_to(stop_pt)
                            self.amr_state = "TRANSFERRING"
                            self.tx_target, self.ty_target = mx, my
                            self.transfer_start = env.now()
                            self.transfer_type = "PICKUP"
                            self.hold(0.6)
                            self.transfer_type = None
                            mach.processed_qty -= 1
                            self.processed_held += 1
                            self.current_target = None
                    else:
                        if self.processed_held > 0:
                            self.amr_state = "MOVING"
                            self.move_to((180, 875))
                            self.amr_state = "TRANSFERRING"
                            self.tx_target, self.ty_target = 130, 875 
                            self.transfer_start = env.now()
                            self.transfer_type = "DELIVER"
                            self.hold(0.6)
                            self.transfer_type = None
                            self.items_handled += self.processed_held
                            self.processed_held = 0
                        else:
                            self.amr_state = "STOPPED"
                            self.hold(0.2)
                else:
                    if self.processed_held > 0:
                        self.amr_state = "MOVING"
                        self.move_to((180, 875))
                        self.amr_state = "TRANSFERRING"
                        self.tx_target, self.ty_target = 130, 875
                        self.transfer_start = env.now()
                        self.transfer_type = "DELIVER"
                        self.hold(0.6)
                        self.transfer_type = None
                        self.items_handled += self.processed_held
                        self.processed_held = 0
                    else:
                        self.amr_state = "STOPPED"
                        self.hold(0.2)

    # Instantiate AMRs and Distribute among areas
    park_slots = []
    for px in park_x_positions_top: park_slots.append((px, 12, True))
    for px in park_x_positions_bottom: park_slots.append((px, 938, False))
    
    current_amr_idx = 0
    for area_idx, count in enumerate(amr_dist):
        area_id = area_idx + 1
        for _ in range(count):
            px, py, is_top = park_slots[current_amr_idx]
            a = AMR(idx=current_amr_idx, px=px, py=py, is_top_parking=is_top, assigned_area=area_id)
            amr_fleets[area_id].append(a)
            all_amrs.append(a)
            current_amr_idx += 1

    # =========================================================================
    # 5. Static UI & Background
    # =========================================================================
    sim.AnimateRectangle(spec=(0, 0, 700, 1000), fillcolor="lightgrey", linecolor="white", linewidth=2)

    # Same color palette used for the AMRs
    area_ui_colors = ["royalblue", "firebrick", "darkorange", "purple", "teal", "magenta", "olive","tan","80%gray","darkturquoise"]

    line_x_positions = [180, 230, 330, 380, 480]
    for x in line_x_positions: sim.AnimateLine([x, 50, x, 915], linewidth=4, linecolor="limegreen")
    line_y_positions = [175, 275, 375, 475, 575, 675, 775]
    for y in line_y_positions: sim.AnimateLine([180, y, 380, y], linewidth=4, linecolor="limegreen")
    line_y_positions = [75, 125, 225, 325, 425, 525, 625, 725, 825]
    for y in line_y_positions:
        sim.AnimateLine([180, y, 230, y], linewidth=4, linecolor="limegreen")
        sim.AnimateLine([330, y, 380, y], linewidth=4, linecolor="limegreen")
    sim.AnimateLine([180, 875, 480, 875], linewidth=4, linecolor="limegreen")
    sim.AnimateLine([480, 40, 480, 50], linewidth=4, linecolor="limegreen")
    sim.AnimateLine([450, 40, 480, 40], linewidth=4, linecolor="limegreen")
    sim.AnimateLine([450, 915, 480, 915], linewidth=4, linecolor="limegreen")

    buffer_configs = [
        {"center_x": 130, "center_y": 275, "text_x": 55, "text_y": 275, "label": "INLET BUFFER"},
        {"center_x": 130, "center_y": 875, "text_x": 55, "text_y": 875, "label": "OUTLET BUFFER"},
    ]
    for buffer in buffer_configs:
        cx, cy = buffer["center_x"], buffer["center_y"]
        tx, ty = buffer["text_x"], buffer["text_y"]
        sim.AnimateRectangle(spec=(cx - 20, cy - 20, cx + 20, cy + 20), fillcolor="green")
        sim.AnimateText(buffer["label"], x=tx, y=ty, textcolor="black", text_anchor="c", fontsize=12, font="bold")

    sim.AnimateRectangle(spec=(150, 0, 450, 50), fillcolor="skyblue")
    sim.AnimateText("AMR STATION 1", x=300, y=40, textcolor="black", text_anchor="c", fontsize=12, font="bold")
    for px in park_x_positions_top: sim.AnimateRectangle(spec=(px - 11, 1, px + 11, 23), fillcolor="salmon")
    
    sim.AnimateRectangle(spec=(150, 900, 450, 950), fillcolor="skyblue")
    sim.AnimateText("AMR STATION 2", x=300, y=920, textcolor="black", text_anchor="c", fontsize=12, font="bold")
    for px in park_x_positions_bottom: sim.AnimateRectangle(spec=(px - 11, 927, px + 11, 949), fillcolor="salmon")

    for i, (mx, my) in enumerate(machine_positions):
        m_area = machines[i].area
        m_color = area_ui_colors[(m_area - 1) % len(area_ui_colors)]
        sim.AnimateRectangle(spec=(mx-20, my-20, mx+20, my+20), fillcolor=m_color, linecolor="white", linewidth=2)

    def state_color(arg, t, idx=0):
        m = machines[idx]
        if m.machine_state == "IDLE": return "yellow"
        if m.machine_state == "BUSY": return "black"
        return "white" 
        
    for i, (mx, my) in enumerate(machine_positions):
        sim.AnimateText(lambda a, t, idx=i: machines[idx].machine_state, x=mx, y=my-5, textcolor=lambda a, t, idx=i: state_color(a, t, idx), text_anchor="c", fontsize=13, font="bold")
        sim.AnimateText(f"M{i+1}", x=mx, y=my+10, textcolor="white", text_anchor="c", fontsize=14, font="bold")

    # =========================================================================
    # 6. Dynamic AMR Render Engine
    # =========================================================================
    def get_x(arg, t, amr):
        if amr.is_moving:
            elapsed = t - amr.move_start_time
            pct = min(1.0, elapsed / max(0.001, amr.move_duration))
            return amr.x_from + pct * (amr.x_to - amr.x_from)
        return amr.x

    def get_y(arg, t, amr):
        if amr.is_moving:
            elapsed = t - amr.move_start_time
            pct = min(1.0, elapsed / max(0.001, amr.move_duration))
            return amr.y_from + pct * (amr.y_to - amr.y_from)
        return amr.y

    def get_transfer_rect(arg, t, amr):
        if not amr.transfer_type: return (-100, -100, -100, -100)
        pct = min(1.0, (t - amr.transfer_start) / 0.6)
        if amr.transfer_type == "DELIVER":
            cx = amr.x + pct * (amr.tx_target - amr.x)
            cy = amr.y + pct * (amr.ty_target - amr.y)
        else:
            cx = amr.tx_target + pct * (amr.x - amr.tx_target)
            cy = amr.ty_target + pct * (amr.y - amr.ty_target)
        return (cx - 4, cy - 4, cx + 4, cy + 4)

    def draw_amr(amr):
        if not amr.active: return
        sim.AnimateRectangle(
            spec=lambda arg, t: (get_x(None, t, amr)-10, get_y(None, t, amr)-10, get_x(None, t, amr)+10, get_y(None, t, amr)+10),
            fillcolor=amr.color, linecolor="cyan", linewidth=2
        )
        sim.AnimateText(
            text=lambda arg, t: f"R:{amr.raw_held} P:{amr.processed_held} Q:{len(amr.request_queue)}",
            x=lambda arg, t: get_x(None, t, amr), y=lambda arg, t: get_y(None, t, amr) + 16,
            textcolor="black", fontsize=14, font="bold", text_anchor="c"
        )
        sim.AnimateText(
            text=lambda arg, t: f"[{amr.items_handled}/20]",
            x=lambda arg, t: get_x(None, t, amr), y=lambda arg, t: get_y(None, t, amr) - 16,
            textcolor="darkblue", fontsize=14, text_anchor="c"
        )
        sim.AnimateRectangle(spec=lambda arg, t: get_transfer_rect(None, t, amr), fillcolor="coral")

    for a in all_amrs:
        draw_amr(a)

    # =========================================================================
    # 7. Dashboard 
    # =========================================================================
    mean_util_monitor = sim.Monitor("Mean Utilisation Rate", level=True)

    class UtilTracker(sim.Component):
        def process(self):
            while True:
                current_time = max(0.001, env.now())
                utils = []
                for mach in machines:
                    live_run = mach.total_run
                    if mach.machine_state == "BUSY":
                        live_run += (current_time - mach.t_start_run)
                    utils.append((live_run / current_time) * 100)
                mean_util = sum(utils) / len(utils) if utils else 0
                mean_util_monitor.tally(mean_util)
                self.hold(1)
                
    UtilTracker() 

    # =========================================================================
    # 8. Aggregated Stats
    # =========================================================================
    def get_all_utils(t):
        utils = []
        for mach in machines:
            live_run = mach.total_run
            if mach.machine_state == "BUSY":
                live_run += (t - mach.t_start_run)
            util = (live_run / max(0.001, t)) * 100
            utils.append(util)
        return utils

    def get_mean_util(arg, t):
        utils = get_all_utils(t)
        mean_val = sum(utils) / len(utils) if utils else 0
        return f"Mean Utilisation: {mean_val:.2f}%"

    def get_std_util(arg, t):
        utils = get_all_utils(t)
        if len(utils) < 2: return "Std Dev: 0.00%"
        mean_val = sum(utils) / len(utils)
        variance = sum((x - mean_val)**2 for x in utils) / len(utils)
        std_val = math.sqrt(variance)
        return f"Std Dev Utilisation: {std_val:.2f}%"

    sim.AnimateText(text=get_mean_util, x=600, y=450, text_anchor="c", fontsize=20, font="bold", textcolor="blue")
    sim.AnimateText(text=get_std_util, x=600, y=425, text_anchor="c", fontsize=20, font="bold", textcolor="blue")

    # =========================================================================
    # 9. Excel Exporter Component
    # =========================================================================
    class DataExporter(sim.Component):
        def process(self):
            self.hold(3600)
            print(f"\n--- Attempting to export data to Excel at t={env.now()} seconds ---")
            times, values = mean_util_monitor.tx()
            df_graph = pd.DataFrame({"Time (s)": times, "Mean Utilisation (%)": values})
            file_name = r"C:\Python\New folder\machine_stats.xlsx"
            
            try:
                with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
                    df_graph.to_excel(writer, sheet_name="Mean Utilisation Trend", index=False)
                    workbook = writer.book
                    worksheet = writer.sheets["Mean Utilisation Trend"]
                    chart = workbook.add_chart({'type': 'line'})
                    max_row = len(df_graph)
                    chart.add_series({
                        'name':       'Mean Utilisation',
                        'categories': ['Mean Utilisation Trend', 1, 0, max_row, 0],
                        'values':     ['Mean Utilisation Trend', 1, 1, max_row, 1],
                        'line':       {'color': 'blue', 'width': 2}
                    })
                    chart.set_title({'name': 'Mean Machine Utilisation Over Time'})
                    chart.set_x_axis({'name': 'Time (Seconds)'})
                    chart.set_y_axis({'name': 'Utilisation (%)', 'min': 0, 'max': 100})
                    worksheet.insert_chart('D2', chart)
                print(f"Success! Data and graph exported to {file_name}\n")
            except Exception as e:
                print(f"\n❌ EXPORT FAILED! Reason: {e}\n")
            env.paused(True)
            
    exporter = DataExporter()

run_simulation()
env.run(env.inf)